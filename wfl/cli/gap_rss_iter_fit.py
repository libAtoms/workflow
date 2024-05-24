#!/usr/bin/env python3

import sys
import os
import warnings

# must do this here to ensure that all ranks except 0 hang and just wait for mpipool tasks
import wfl.autoparallelize.mpipool_support

wfl.autoparallelize.mpipool_support.init()

import glob
import json
import pathlib
import pprint
from datetime import datetime
from pathlib import Path
import re

import ase.atoms
import click
import numpy as np
import yaml
try:
    from quippy.potential import Potential
except ModuleNotFoundError:
    pass


import wfl
import wfl.descriptors.quippy
import wfl.descriptor_heuristics
import wfl.fit.error
from wfl.fit.gap import multistage as gap_multistage
import wfl.generate.atoms_and_dimers
import wfl.generate.buildcell
import wfl.select.by_descriptor
import wfl.select.convex_hull
import wfl.select.simple
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import md, optimize, supercells
from wfl.select.flat_histogram import biased_select_conf
from wfl.select.selection_space import val_relative_to_nearby_composition_volume_min
from wfl.descriptor_heuristics import descriptors_from_length_scales
from wfl.utils.params import Params
from wfl.utils.version import get_wfl_version
from wfl.utils.misc import dict_tuple_keys_to_str
from wfl.calculators import generic
from wfl.calculators.vasp import Vasp


@click.group()
@click.option('--verbose', '-v', is_flag=True)
@click.option('--configuration', '-c', type=click.STRING, required=True)
@click.option('--buildcell_cmd', '-b', type=click.STRING, default='buildcell', envvar='GRIF_BUILDCELL_CMD')
@click.option('--cur_iter', '-i', type=click.INT, default=None)
@click.option('--seeds', '-s', type=click.STRING, default=None)
@click.pass_context
def cli(ctx, verbose, configuration, buildcell_cmd, cur_iter, seeds):
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['buildcell_cmd'] = buildcell_cmd
    ctx.obj['cur_iter'] = cur_iter

    ctx.obj['seeds'] = seeds
    if ctx.obj['seeds'] is not None:
        warnings.warn(f'Setting initial seeds to {ctx.obj["seeds"]}.  If autoparallelization '
                       'is on (WFL_NUM_PYTHON_SUBPROCESSES, etc), this will not affect runs on other threads.')
        ctx.obj['seeds'] = [int(s) for s in ctx.obj['seeds'].split(',')]
        ctx.obj['rng'] = np.random.default_rng(ctx.obj['seeds'][0])
        del ctx.obj['seeds'][0]

    print_log('GAP_RSS_ITER_FIT STARTING, code version ' + get_wfl_version(), blank_lines=True)

    config = json.load(open(configuration))

    # gather all chemical species across all compositions
    Zs = set()
    for formula, _ in config['global']['compositions']:
        numbers = ase.atoms.Atoms(formula).numbers
        Zs |= set(numbers)
    Zs = sorted(Zs)
    config['global']['Zs'] = Zs

    # figure out actual compositions
    compositions = []
    for formula, frac in config['global']['compositions']:
        numbers = ase.atoms.Atoms(formula).numbers
        # must convert to int since leaving it as np integer breaks yaml save/restore
        composition = [int(sum(numbers == Z)) for Z in Zs]
        compositions.append((composition, frac))

    frac_tot = sum(c[1] for c in compositions)
    compositions = dict({(tuple(c[0]), c[1] / frac_tot) for c in compositions})
    config['global']['compositions'] = compositions
    if verbose:
        print('compositions', config['global']['compositions'])

    ctx.obj['params'] = Params(config)


def Z_label(Zs, c_inds):
    return '__'.join(['{}_{}'.format(Z, n) for (Z, n) in zip(Zs, c_inds) if n != 0])


def is_elemental(c):
    return bool(sum(np.array(c) != 0) == 1)


def print_log(msg, show_time=True, blank_lines=False, logfiles=[sys.stdout, sys.stderr]):
    if show_time:
        time_str = ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        time_str = ''

    for logf in logfiles:
        if blank_lines:
            logf.write('\n')
        for l in msg.splitlines():
            logf.write('LOG' + time_str + ': ' + l + '\n')
        if blank_lines:
            logf.write('\n')
        logf.flush()


@cli.command('prep')
@click.option('--length-scales-file', type=click.STRING, help='length scales file')
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def prep(ctx, length_scales_file, verbose):
    print_log('GAP_RSS_ITER_FIT DOING STEP prep', blank_lines=True)

    verbose = verbose or ctx.obj['verbose']

    params = ctx.obj['params']

    # elements and compositions
    Zs = np.array(params.get('global/Zs'), dtype=int)
    compositions = params.get('global/compositions')
    print_log('Zs ' + pprint.pformat(Zs, indent=2))
    print('compositions\n' + pprint.pformat(compositions, indent=2))

    # volume
    volume_factor = params.get('global/volume_factor', default=1.0)
    length_factor = volume_factor ** (1.0 / 3.0)
    print('volume_factor', volume_factor)

    # length scales
    if length_scales_file is None:
        with open(params.get('prep/length_scales_file')) as fin:
            length_scales = yaml.safe_load(fin)
    else:
        with open(length_scales_file) as fin:
            length_scales = yaml.safe_load(length_scales_file)
    if any([Z not in length_scales for Z in Zs]):
        raise RuntimeError('One of elements {} not in length scales {}'.format(Zs, list(length_scales)))
    print('length_scales\n' + pprint.pformat(length_scales, indent=2))

    # prep buildcell inputs using Zs, compositions, length scales
    for buildcell_step_type, natom in params.get('prep/buildcell', {'default': [6, 24]}).items():
        buildcell_inputs = {}
        for c_inds in compositions:
            if verbose:
                print('prep creating buildcell input c_inds', c_inds)
            buildcell_inputs[c_inds] = {}
            if is_elemental(c_inds):
                # elemental
                Z_elem = Zs[np.where(np.array(c_inds) != 0)[0][0]]

                f = f'buildcell.narrow_vol_range_even.Z_{Z_elem}.{buildcell_step_type}.input'
                buildcell_inputs[c_inds]['narrow_even'] = [f, 0.4]
                s = wfl.generate.buildcell.create_input(
                    z=Z_elem, vol_per_atom=volume_factor * length_scales[Z_elem]['vol_per_atom'][0],
                    bond_lengths=length_factor * length_scales[Z_elem]['bond_len'][0],
                    natom=natom, verbose=verbose)
                with open(f, "w") as fout:
                    fout.write(s + "\n")

                f = f'buildcell.narrow_vol_range_odd.Z_{Z_elem}.{buildcell_step_type}.input'
                buildcell_inputs[c_inds]['narrow_odd'] = [f, 0.1]
                s = wfl.generate.buildcell.create_input(
                    z=Z_elem, vol_per_atom=volume_factor * length_scales[Z_elem]['vol_per_atom'][0],
                    bond_lengths=length_factor * length_scales[Z_elem]['bond_len'][0], odd='only',
                    natom=natom, verbose=verbose)
                with open(f, "w") as fout:
                    fout.write(s + "\n")

                f = f'buildcell.wide_vol_range_even.Z_{Z_elem}.{buildcell_step_type}.input'
                buildcell_inputs[c_inds]['wide_even'] = [f, 0.5]
                s = wfl.generate.buildcell.create_input(
                    z=Z_elem, vol_per_atom=volume_factor * length_scales[Z_elem]['vol_per_atom'][0],
                    bond_lengths=length_factor * length_scales[Z_elem]['bond_len'][0], vol_range=(0.75, 1.25),
                    natom=natom, verbose=verbose)
                with open(f, "w") as fout:
                    fout.write(s + "\n")
            else:
                # multicomponent
                Z_label_str = 'Z_' + Z_label(Zs, c_inds)
                used_Zs = [Z for (Z, n) in zip(Zs, c_inds) if n != 0]
                used_composition = [n for n in c_inds if n != 0]

                f = f'buildcell.narrow_vol_range.{Z_label_str}.{buildcell_step_type}.input'
                buildcell_inputs[c_inds]['narrow'] = [f, 0.5]
                s = wfl.generate.buildcell.create_input(
                    z=used_Zs, composition=used_composition,
                    vol_per_atom=[volume_factor * length_scales[Z]['vol_per_atom'][0] for Z in used_Zs],
                    bond_lengths=[length_factor * length_scales[Z]['bond_len'][0] for Z in used_Zs], odd='also',
                    natom=natom, verbose=verbose)
                with open(f, "w") as fout:
                    fout.write(s + "\n")

                f = f'buildcell.wide_vol_range.{Z_label_str}.{buildcell_step_type}.input'
                buildcell_inputs[c_inds]['wide'] = [f, 0.5]
                s = wfl.generate.buildcell.create_input(
                    z=used_Zs, composition=used_composition,
                    vol_per_atom=[volume_factor * length_scales[Z]['vol_per_atom'][0] for Z in used_Zs],
                    bond_lengths=[length_factor * length_scales[Z]['bond_len'][0] for Z in used_Zs], vol_range=(0.75, 1.25), odd='also',
                    natom=natom, verbose=verbose)
                with open(f, "w") as fout:
                    fout.write(s + "\n")

        print('buildcell_inputs\n' + pprint.pformat(buildcell_inputs, indent=2))
        buildcell_input_file_stem = 'gap_rss_iter_fit.prep.buildcell_inputs'
        if len(buildcell_step_type) > 0:
            buildcell_input_file_stem += '.' + buildcell_step_type
        with open(buildcell_input_file_stem + '.yaml', 'w') as fout:
            yaml.dump(buildcell_inputs, stream=fout, default_flow_style=False)

    # prep GAP fitting config using Zs, length scales
    fit_params = gap_multistage.prep_params(Zs, length_scales, params.get('fit/GAP_template_file'),
                                            sharpness=params.get('fit/universal_SOAP_sharpness', default=0.5))
    yaml.dump(fit_params, open('multistage_GAP_fit_settings.yaml', 'w'), indent=4)

    # similarly prep config-selection descriptor using Zs, length scales
    config_select_descriptor = params.get('global/config_selection_descriptor')
    config_selection_descs, desc_Zs = descriptors_from_length_scales(
        config_select_descriptor, Zs, length_scales)
    descs_by_Z = {}
    for Zcenter in Zs:
        # convert key to plain int so yaml dump will be readable with safe_load
        descs_by_Z[int(Zcenter)] = [desc for ZZ, desc in zip(desc_Zs, config_selection_descs) if ZZ == Zcenter]

    with open('gap_rss_iter_fit.prep.config_selection_descriptors.yaml', 'w') as fout:
        yaml.dump(descs_by_Z, stream=fout, default_flow_style=False)

    kwargs = {}
    dimer_n_steps = params.get('prep/dimer_n_steps', default=None)
    if dimer_n_steps is not None:
        kwargs['dimer_n_steps'] = dimer_n_steps
    # should this really overwrite atoms_and_dimers.xyz?
    wfl.generate.atoms_and_dimers.prepare(OutputSpec('atoms_and_dimers.xyz'),
                                                  Zs, {Z: length_scales[Z]['min_bond_len'][0] for Z in Zs},
                                                  max_cutoff=gap_multistage.max_cutoff(fit_params),
                                                  **kwargs)


def create_all_buildcell(cur_iter, run_dir, Zs, compositions, N_configs_tot,
                         buildcell_cmd, buildcell_input_files, buildcell_pert,
                         single_composition_group, verbose=False):

    groups = {}

    config_i_start = 0
    # call buildcell
    for compos_inds, compos_frac in compositions.items():
        N_configs_compos = N_configs_tot * compos_frac
        print('Using buildcell to create random initial configs for composition',
              compos_inds, 'fraction', compos_frac, 'N_configs_compos', N_configs_compos)

        buildcell_inp_of_compos = buildcell_input_files[compos_inds]

        label_str = 'Z_' + Z_label(Zs, compos_inds)
        compos_structs = []
        # assemble input strings and corresponding numbers of configs
        for buildcell_type, buildcell_filename_fract in buildcell_inp_of_compos.items():
            with open(buildcell_filename_fract[0]) as fin:
                buildcell_input = fin.read()
            buildcell_fract = buildcell_filename_fract[1]
            N_configs = int(N_configs_compos * buildcell_fract)
            if verbose:
                print('buildcell_type', buildcell_type, buildcell_fract, 'N_configs', N_configs)
            if N_configs <= 0:
                raise ValueError(f'Total number of configurations {N_configs_tot} * fraction requested for this '
                                 f'composition {compos_frac:.3f} and variety {buildcell_fract:.3f} gives 0')

            # output_force is set here (and below) so that it will not fail even if this has run before
            # because actual operation will skip it in that case.
            c_out = OutputSpec(f'initial_random_configs.{label_str}.{buildcell_type}.xyz', file_root=run_dir)
            extra_info = {'buildcell_type': buildcell_type}
            if single_composition_group:
                extra_info['gap_rss_group'] = 'ALL'
            else:
                extra_info['gap_rss_group'] = label_str
            extra_info['gap_rss_iter'] = cur_iter
            structs = wfl.generate.buildcell.buildcell(range(config_i_start, config_i_start + N_configs), c_out,
                                                 buildcell_cmd=buildcell_cmd, buildcell_input=buildcell_input,
                                                 extra_info=extra_info, perturbation=buildcell_pert,
                                                 verbose=verbose)

            compos_structs.append(structs)
            config_i_start += N_configs

        print('merging buildcell types for this composition', compos_structs)
        groups[label_str] = {'cur_confs': ConfigSet(compos_structs), 'frac': compos_frac}

    if single_composition_group:
        # merge groups into one
        print('merging composition groups', groups.values())
        groups = {'ALL': {'cur_confs': ConfigSet([grp['cur_confs'] for grp in groups.values()]),
                          'frac': 1.0}}
        if verbose:
            print('got combined groups', groups)

    return groups


def process_cur_iter(cur_iter):
    if cur_iter is None:
        # read from file
        try:
            with open('ACTIVE_ITER') as fin:
                cur_iter = int(fin.readline())
        except FileNotFoundError:
            # initialize file
            cur_iter = 0
            with open('ACTIVE_ITER', 'w') as fout:
                fout.write('{}\n'.format(cur_iter))

    return cur_iter


def increment_active_iter(cur_iter):
    try:
        with open('ACTIVE_ITER') as fin:
            file_cur_iter = int(fin.readline())
    except FileNotFoundError:
        file_cur_iter = None
    if file_cur_iter is not None and cur_iter + 1 > file_cur_iter:
        # file exists and incrementing past its value
        with open('ACTIVE_ITER', 'w') as fout:
            fout.write('{}\n'.format(cur_iter + 1))


def evaluate_ref(dft_in_configs, dft_evaluated_configs, params, run_dir, verbose=False):
    """Do reference property evaluation, parameters taken from system config dict

    Parameters
    ----------
    dft_in_configs : ConfigSet
    dft_evaluated_configs : OutputSpec
    params : Params
        run parameters
    run_dir
    verbose : bool, default False

    Returns
    -------
    evaluated_configs : ConfigSet
        configurations with evaluated reference quantities
    """

    if verbose:
        keep_files = "default"
    else:
        keep_files = False

    if params.dft_code == "VASP":
        calculator = Vasp
    else:
        raise ValueError(f"Unsupported dft_code {params.dft_code}")

    return generic.calculate(
        inputs=dft_in_configs,
        outputs=dft_evaluated_configs,
        calculator=calculator(workdir=run_dir, keep_files=keep_files, **params.dft_params.get("kwargs", {})),
        output_prefix="REF_",
        autopara_info={'remote_label': 'REF_eval'}
    )


def get_old_fitting_files(cur_iter, extra_fitting_files=[]):
    old_fitting_files = []
    for prev_iter in range(cur_iter):
        old_fitting_files.extend(
            glob.glob(os.path.join('run_iter_{}'.format(prev_iter), 'DFT_evaluated_fitting.*.xyz')))
    old_fitting_files += extra_fitting_files

    return old_fitting_files


def do_fit_and_test(cur_iter, run_dir, params, fitting_configs, testing_configs=None,
                    database_modify_mod=None, seeds=None, verbose=False):
    # modify database if needed
    if database_modify_mod is not None:
        # load configs into memory so they can be modified
        fitting_configs = ConfigSet(list(fitting_configs))
        import importlib
        database_modify_mod = importlib.import_module(database_modify_mod)
        database_modify_mod.modify(fitting_configs)

    print_log('fitting')
    # fit
    with open('multistage_GAP_fit_settings.yaml') as fin:
        fit_params = yaml.safe_load(fin)
    GAP_xml_file, GAP_name = gap_multistage.fit(fitting_configs, GAP_name=f'GAP_iter_{cur_iter}', params=fit_params,
                                                seeds=seeds, skip_if_present=True, run_dir=run_dir, verbose=verbose)

    calculator = (Potential, [], {'param_filename': GAP_xml_file, 'args_str': f'Potential xml_label={GAP_name}'})

    if params.get('fit/calc_fitting_error', default=True):
        co = OutputSpec(f'fitting.error_database.GAP_iter_{cur_iter}.xyz', file_root=run_dir)
        for at in fitting_configs:
            at.calc = None
        evaluated_configs = generic.calculate(fitting_configs, co, calculator, output_prefix="GAP_")
        fitting_error = wfl.fit.error.calc(evaluated_configs, calc_property_prefix="GAP_", ref_property_prefix="REF_",
                category_keys=['config_type', 'gap_rss_iter'])
        with open(GAP_xml_file + '.fitting_err.json', 'w') as fout:
            json.dump(dict_tuple_keys_to_str(fitting_error[0]), fout)
        print('FITTING ERROR')
        pprint.pprint(fitting_error)
    if testing_configs is not None:
        co = OutputSpec(f'testing.error_database.GAP_iter_{cur_iter}.xyz', file_root=run_dir)
        for at in testing_configs:
            at.calc = None
        evaluated_configs = generic.calculate(testing_configs, co, calculator, output_prefix="GAP_")
        testing_error = wfl.fit.error.calc(evaluated_configs, calc_property_prefix="GAP_", ref_property_prefix="REF_",
                category_keys=['config_type', 'gap_rss_iter'])
        with open(GAP_xml_file + '.testing_err.json', 'w') as fout:
            json.dump(dict_tuple_keys_to_str(testing_error[0]), fout)
        print('TESTING ERROR')
        pprint.pprint(testing_error)

    return GAP_xml_file


def evaluate_iter_and_fit_all(cur_iter, run_dir, params, step_params, cur_fitting_configs, testing_configs,
                              database_modify_mod, calc_fitting_error, extra_fitting_files=[],
                              seeds=None, verbose=False):
    # code below is ugly mess of combining things with files, lists and ConfigSet - probably indicates
    # some design flaw someplace

    print_log('evaluating with DFT')
    # evaluate fitting configs with DFT
    fitting_configs = ConfigSet(cur_fitting_configs)
    fitting_configs_out = OutputSpec('DFT_evaluated_fitting.ALL.xyz', file_root=run_dir)
    evaluated_configs = evaluate_ref(fitting_configs, fitting_configs_out, params, run_dir, verbose)

    error_scale_factor = step_params.get('fit_error_scale_factor', None)
    if error_scale_factor is not None:
        # add fit_error_scale_factor to every config's Atoms.info dict
        co = OutputSpec("DFT_evaluated_fitting.error_scale_factor.ALL.xyz", file_root=run_dir)
        if not co.all_written():
            for at in evaluated_configs:
                at.info["fit_error_scale_factor"] = error_scale_factor
                co.store(at)
            co.close()
        evaluated_configs = ConfigSet(co)

    fitting_configs = [evaluated_configs]
    # gather old fitting files
    old_fitting_files = get_old_fitting_files(cur_iter, extra_fitting_files)
    if len(old_fitting_files) > 0:
        fitting_configs += [ConfigSet(old_fitting_files)]
    # Only configsets from the same source can be merged like this
    # so we are implicitly relying on evaluate_ref to return a configset
    # that is file based (because we added a ConfigSet based on old_fitting_files,
    # which are definitely files), which might in principle be a problem.
    fitting_configs = ConfigSet(fitting_configs)

    # evaluate testing configs (if any) with DFT
    if any([c is not None for c in testing_configs]):
        testing_configs = ConfigSet([c for c in testing_configs if c is not None])
        testing_configs_out = OutputSpec('DFT_evaluated_testing.ALL.xyz', file_root=run_dir)
        testing_configs = [evaluate_ref(testing_configs, testing_configs_out, params, run_dir, verbose)]
    else:
        testing_configs = []
    # combine with previous iters testing configs, if any
    old_testing_files = []
    for prev_iter in range(cur_iter):
        old_testing_files.extend(
            glob.glob(os.path.join('run_iter_{}'.format(prev_iter), 'DFT_evaluated_testing.*.xyz')))
    if len(old_testing_files) > 0:
        testing_configs += [ConfigSet(old_testing_files)]
    if len(testing_configs) > 0:
        testing_configs = ConfigSet(testing_configs)
    else:
        testing_configs = None

    GAP_xml_file = do_fit_and_test(cur_iter, run_dir, params, fitting_configs, testing_configs,
                                   database_modify_mod=database_modify_mod, seeds=seeds, verbose=verbose)
    return GAP_xml_file


def get_buildcell_input_files(step_label, cur_iter):
    files = (glob.glob(f"gap_rss_iter_fit.prep.buildcell_inputs.{step_label}.iter_*.yaml") +
             glob.glob(f"gap_rss_iter_fit.prep.buildcell_inputs.{step_label}.yaml") +
             glob.glob("gap_rss_iter_fit.prep.buildcell_inputs.default.iter_*.yaml") +
             glob.glob("gap_rss_iter_fit.prep.buildcell_inputs.default.yaml"))
    if len(files) == 0:
        raise RuntimeError(f"No buildcell inputs yaml file for step {step_label}")

    use_filename = None

    # look for iter specific file
    for filename in files:
        m = re.search(r"\.iter_([0-9:]+)\.yaml$", filename)
        if not m:
            # not iter specific, skip for now
            continue
        file_iter_range = m.group(1).split(":")
        if len(file_iter_range) == 1:
            if cur_iter != int(file_iter_range[0]):
                continue
        elif len(file_iter_range) == 2:
            if ((len(file_iter_range[0]) != 0 and cur_iter < int(file_iter_range[0])) or
                (len(file_iter_range[1]) != 0 and cur_iter >= int(file_iter_range[1]))):
                continue
        else:
            raise RuntimeError("buildcell inputs yaml filename range expression {m.group(1)} not valid")

        use_filename = filename
        break

    if use_filename is None:
        # look for non-iter-specific file
        for filename in files:
            if filename.endswith(f".{step_label}.yaml") or filename.endswith(".default.yaml"):
                use_filename = filename
                break

    with open(use_filename) as fin:
        # must use full load because dict keys are tuples
        buildcell_input_files = yaml.full_load(fin)

    return buildcell_input_files


@cli.command('initial_step')
@click.option('--cur_iter', '-i', type=click.INT, default=None)
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def do_initial_step(ctx, cur_iter, verbose):
    verbose = verbose or ctx.obj['verbose']
    cur_iter = process_cur_iter(ctx.obj['cur_iter'] if cur_iter is None else cur_iter)
    print_log(f'GAP_RSS_ITER_FIT DOING STEP initial_step {cur_iter}', blank_lines=True)

    params = ctx.obj['params']
    params.cur_iter = cur_iter
    run_dir, Zs, compositions = step_startup(params, cur_iter)

    N_configs_tot = params.get('initial_step/buildcell_total_N')
    buildcell_pert = params.get('initial_step/buildcell_pert', default=0.1)
    single_composition_group = params.get('global/single_composition_group', default=True)

    select_by_desc_method = params.get('global/select_by_desc_method', default='CUR')

    buildcell_cmd = ctx.obj['buildcell_cmd']
    buildcell_input_files = get_buildcell_input_files("initial", cur_iter)

    # SAVE gap_rss_group in Atoms.info for later steps like doing convex hull
    #     based sigma setting separately for each group's convex hull

    # store info for each group: fraction of total, and configs
    groups = create_all_buildcell(cur_iter, run_dir, Zs, compositions, N_configs_tot, buildcell_cmd,
                                  buildcell_input_files, buildcell_pert, single_composition_group, verbose)

    # calculate descriptors
    with open('gap_rss_iter_fit.prep.config_selection_descriptors.yaml') as fin:
        descriptor_strs = yaml.safe_load(fin)

    print_log('selecting config by descriptor')
    select_fitting_and_testing_for_groups(run_dir, cur_iter, groups, Params(params.get('initial_step'), cur_iter), Zs,
                                          None, select_by_desc_method, descriptor_strs,
                                          params.get('global/config_selection_descriptor_local', default=False),
                                          flat_histo=False, rng=ctx.obj['rng'], verbose=verbose)

    atoms_dimers = ConfigSet('atoms_and_dimers.xyz')

    _ = evaluate_iter_and_fit_all(cur_iter, run_dir, params, Params(params.get('initial_step'), cur_iter),
                                  [grp['cur_confs'] for grp in groups.values()] + [atoms_dimers],
                                  [grp['testing_confs'] for grp in groups.values()],
                                  params.get('fit/database_modify_mod'),
                                  params.get('fit/calc_fitting_error', default=True),
                                  extra_fitting_files=params.get('fit/extra_fitting_files', default=[]),
                                  seeds=ctx.obj['seeds'], verbose=verbose)

    print_log('done')
    increment_active_iter(cur_iter)


@cli.command('rss_step')
@click.option('--cur_iter', type=click.INT, default=None)
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def do_rss_step(ctx, cur_iter, verbose):
    verbose = verbose or ctx.obj['verbose']
    cur_iter = process_cur_iter(ctx.obj['cur_iter'] if cur_iter is None else cur_iter)
    print_log(f'GAP_RSS_ITER_FIT DOING STEP rss_step {cur_iter}', blank_lines=True)

    params = ctx.obj['params']
    params.cur_iter = cur_iter
    run_dir, Zs, compositions = step_startup(params, cur_iter)

    N_configs_tot = params.get('rss_step/buildcell_total_N')
    buildcell_pert = params.get('rss_step/buildcell_pert', default=0.0)
    single_composition_group = params.get('global/single_composition_group', default=True)
    traj_select_by_desc_method = params.get('global/prelim_select_by_desc_method', default='CUR')
    select_by_desc_method = params.get('global/select_by_desc_method', default='CUR')

    buildcell_cmd = ctx.obj['buildcell_cmd']
    buildcell_input_files = get_buildcell_input_files("rss", cur_iter)
    with open('gap_rss_iter_fit.prep.config_selection_descriptors.yaml') as fin:
        descriptor_strs = yaml.safe_load(fin)

    # store info for each group: fraction of total, and configs
    # NEED TO SAVE gap_rss_group in Atoms.info for later steps like fitting convex hull
    #     based sigma setting separately for each group's convex hull
    groups = create_all_buildcell(cur_iter, run_dir, Zs, compositions, N_configs_tot, buildcell_cmd,
                                  buildcell_input_files,
                                  buildcell_pert, single_composition_group, verbose)

    prev_GAP = os.path.join('run_iter_{}'.format(cur_iter - 1), 'GAP_iter_{}.xml'.format(cur_iter - 1))

    RSS_minima_diverse(run_dir, groups, Params(params.get('rss_step'), cur_iter), Zs,
                       traj_select_by_desc_method, descriptor_strs,
                       params.get('global/config_selection_descriptor_local', default=False),
                       prev_GAP, select_convex_hull=params.get('rss_step/select_convex_hull'),
                       get_entire_trajectories=True, optimize_kwargs=params.get('rss_step/optimize_kwargs', {}),
                       rng=ctx.obj['rng'], verbose=verbose)

    select_fitting_and_testing_for_groups(run_dir, cur_iter, groups, Params(params.get('rss_step'), cur_iter), Zs,
                                          'last_op__optimize_energy', select_by_desc_method, descriptor_strs,
                                          params.get('global/config_selection_descriptor_local', default=False),
                                          rng=ctx.obj['rng'], verbose=verbose)

    _ = evaluate_iter_and_fit_all(cur_iter, run_dir, params, Params(params.get('rss_step'), cur_iter),
                                  [grp['cur_confs'] for grp in groups.values()],
                                  [grp['testing_confs'] for grp in groups.values()],
                                  params.get('fit/database_modify_mod'),
                                  params.get('fit/calc_fitting_error', default=True),
                                  extra_fitting_files=params.get('fit/extra_fitting_files', default=[]),
                                  seeds=ctx.obj['seeds'], verbose=verbose)

    print_log('done')
    increment_active_iter(cur_iter)


@cli.command('MD_bulk_defect_step')
@click.option('--cur_iter', type=click.INT, default=None)
@click.option('--minima_file', default=None)
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def do_MD_bulk_defect_step(ctx, cur_iter, minima_file, verbose):
    verbose = verbose or ctx.obj['verbose']
    cur_iter = process_cur_iter(ctx.obj['cur_iter'] if cur_iter is None else cur_iter)
    print_log(f'GAP_RSS_ITER_FIT DOING STEP MD_bulk_defect_step {cur_iter}', blank_lines=True)

    params = ctx.obj['params']
    params.cur_iter = cur_iter
    run_dir, Zs, compositions = step_startup(params, cur_iter)

    single_composition_group = params.get('global/single_composition_group', default=True)
    minima_select_by_desc_method = params.get('MD_bulk_defect_step/prelim_select_by_desc_method',
                                              default=params.get('global/prelim_select_by_desc_method', default='CUR'))
    select_by_desc_method = params.get('MD_bulk_defect_step/select_by_desc_method',
                                       default=params.get('global/select_by_desc_method', default='CUR'))
    with open('gap_rss_iter_fit.prep.config_selection_descriptors.yaml') as fin:
        descriptor_strs = yaml.safe_load(fin)

    prev_GAP = os.path.join('run_iter_{}'.format(cur_iter - 1), 'GAP_iter_{}.xml'.format(cur_iter - 1))
    optimize_kwargs = params.get('MD_bulk_defect_step/optimize_kwargs', {})

    if minima_file is None:
        # do another round of RSS for minima only
        N_configs_tot = params.get('MD_bulk_defect_step/buildcell_total_N')
        buildcell_pert = params.get('MD_bulk_defect_step/buildcell_pert', default=0.0)
        single_composition_group = params.get('global/single_composition_group', default=True)

        buildcell_cmd = ctx.obj['buildcell_cmd']
        buildcell_input_files = get_buildcell_input_files("MD_bulk_defect", cur_iter)

        # store info for each group: fraction of total, and configs
        # NEED TO SAVE gap_rss_group in Atoms.info for later steps like fitting convex hull
        #     based sigma setting separately for each group's convex hull
        groups = create_all_buildcell(cur_iter, run_dir, Zs, compositions, N_configs_tot, buildcell_cmd,
                                      buildcell_input_files,
                                      buildcell_pert, single_composition_group, verbose)

        RSS_minima_diverse(run_dir, groups, Params(params.get('MD_bulk_defect_step'), cur_iter), Zs,
                           minima_select_by_desc_method, descriptor_strs,
                           params.get('global/config_selection_descriptor_local', default=False),
                           prev_GAP, select_convex_hull=False, get_entire_trajectories=False,
                           optimize_kwargs=optimize_kwargs, rng=ctx.obj['rng'], verbose=verbose)
    else:
        # this will not try to preserve group structure
        groups = {'ALL': {'cur_confs': ConfigSet([minima_file]), 'frac': 1.0}}

    # cell params
    max_n_atoms = params.get('MD_bulk_defect_step/max_n_atoms')
    surf_min_thickness = params.get('MD_bulk_defect_step/surface_min_thickness', 8.0)
    surf_vacuum = params.get('MD_bulk_defect_step/surface_vacuum', 8.0)
    # MD params
    MD_dt = params.get('MD_bulk_defect_step/MD_dt')
    bulk_MD_n_steps = params.get('MD_bulk_defect_step/bulk_MD_n_steps')
    defect_MD_n_steps = params.get('MD_bulk_defect_step/bulk_MD_n_steps')
    # NOTE: might there be a way to guess the relevant temperature range
    # (e.g. 2*T_melt for bulk, 0.75*T_melt for defect) from some energies we already
    # calculated?  Maybe an MD that doesn't end until it detects melting,
    # perhaps on the lowest energy (i.e. most stable) minimum?
    bulk_MD_T_range = params.get('MD_bulk_defect_step/bulk_MD_T_range')
    defect_MD_T_range = params.get('MD_bulk_defect_step/defect_MD_T_range')

    # create supercells
    print_log('creating supercells')
    for grp_label in groups:
        n_bulk_MD = int(params.get('MD_bulk_defect_step/N_bulk', 0) * groups[grp_label]['frac'])

        # go through configs, reading from file if necessary, to get number so that rng.choice()
        # selection below doesn't have to do it repeatedly
        n_minima = len([None for at in groups[grp_label]['cur_confs']])

        minima_inds = ctx.obj['rng'].choice(range(n_minima), n_bulk_MD)
        selected_minima = wfl.select.simple.by_index(
            groups[grp_label]['cur_confs'],
            OutputSpec(f'MD_minima.bulk.{grp_label}.xyz', file_root=run_dir),
            minima_inds)
        groups[grp_label]['bulk_confs'] = supercells.largest_bulk(
            selected_minima,
            OutputSpec(f'MD_cells.bulk.{grp_label}.xyz', file_root=run_dir),
            max_n_atoms=max_n_atoms)

        n_vacancy_MD = int(params.get('MD_bulk_defect_step/N_vacancy', 0) * groups[grp_label]['frac'])
        n_antisite_MD = int(params.get('MD_bulk_defect_step/N_antisite', 0) * groups[grp_label]['frac'])
        n_interstitial_MD = int(params.get('MD_bulk_defect_step/N_interstitial', 0) * groups[grp_label]['frac'])
        n_surface_MD = int(params.get('MD_bulk_defect_step/N_surface', 0) * groups[grp_label]['frac'])

        vacancy_type_args = params.get('MD_bulk_defect_step/vacancy_type_args', [('', {})])
        antisite_type_args = params.get('MD_bulk_defect_step/antisite_type_args', [('', {})])

        defect_confs = []
        for base_label, sc_func, n_configs, sc_extra_args in [('vacancy', supercells.vacancy, n_vacancy_MD, vacancy_type_args),
                                                              ('antisite', supercells.antisite, n_antisite_MD, antisite_type_args),
                                                              ('interstitial', supercells.interstitial, n_interstitial_MD, [('', {})]),
                                                              ('surface', supercells.surface, n_surface_MD,
                                                               [('', {'min_thickness': surf_min_thickness, 'vacuum': surf_vacuum})])]:
            if n_configs <= 0:
                continue
            for extra_label, extra_kwargs in sc_extra_args:
                minima_inds = ctx.obj['rng'].choice(range(n_minima), n_configs)
                label = base_label
                if len(extra_label) > 0:
                    label += '.' + extra_label
                selected_minima = wfl.select.simple.by_index(
                    groups[grp_label]['cur_confs'],
                    OutputSpec(f'MD_minima.{label}.{grp_label}.xyz', file_root=run_dir), minima_inds)

                defect_confs.append(sc_func(selected_minima, OutputSpec(f'MD_cells.{label}.{grp_label}.xyz',
                                                                        file_root=run_dir),
                                            max_n_atoms=max_n_atoms, rng=ctx.obj['rng'], **extra_kwargs))
        # NOTE: grouping all the defect configurations this way makes for better potential parallelism
        # since all the MDs can run side by side, but possibly worse restart for interrupted jobs, since the
        # results are all-or-none on the entire set.
        groups[grp_label]['defect_confs'] = ConfigSet(defect_confs)

    # NOTE: perhaps the hard-wiring of specific md.sample parameters should be replaced with an
    # 'md_kwargs' param, similar to 'optimize_kwargs'
    MD_pressure = params.get('MD_bulk_defect_step/MD_pressure', default=('info', 'optimize_pressure_GPa'))
    optimize_kwargs['pressure'] = MD_pressure
    print_log('doing optimize + MD')
    for grp_label in groups:
        print_log('  group ' + grp_label)
        # MD for bulks
        bulk_traj = md.md(groups[grp_label]['bulk_confs'],
                              OutputSpec(f'bulk_MD_trajs.{grp_label}.xyz', file_root=run_dir),
                              calculator=(Potential, None, {'param_filename': prev_GAP}),
                              steps=bulk_MD_n_steps, dt=MD_dt,
                              temperature=bulk_MD_T_range, temperature_tau=10.0 * MD_dt,
                              pressure=MD_pressure, traj_step_interval=25, rng=ctx.obj['rng'])

        if params.get('MD_bulk_defect_step/optimize_before_MD', default=True):
            # minim defects
            defect_optimize_trajs = optimize.optimize(groups[grp_label]['defect_confs'],
                                           OutputSpec(f'defect_optimize_trajs.{grp_label}.xyz',
                                                      file_root=run_dir),
                                           calculator=(Potential, None, {'param_filename': prev_GAP}),
                                           precon='ID', keep_symmetry=True, rng=ctx.obj['rng'], **optimize_kwargs)
            defect_starting = wfl.select.simple.by_bool_func(defect_optimize_trajs,
                                                             OutputSpec(f'defect_minima.{grp_label}.xyz',
                                                                        file_root=run_dir),
                                                             lambda at: at.info["optimize_config_type"].startswith("optimize_last"))
        else:
            defect_starting = groups[grp_label]['defect_confs']

        # MD defects
        defect_traj = md.md(defect_starting,
                                OutputSpec(f'defect_MD_trajs.{grp_label}.xyz', file_root=run_dir),
                                calculator=(Potential, None, {'param_filename': prev_GAP}),
                                steps=defect_MD_n_steps, dt=MD_dt,
                                temperature=defect_MD_T_range, temperature_tau=10.0 * MD_dt,
                                pressure=MD_pressure, traj_step_interval=50, rng=ctx.obj['rng'])

        groups[grp_label]['cur_confs'] = ConfigSet([bulk_traj, defect_traj])

    print_log('selecting by flat histogram + by-descriptor from MD trajectories')
    select_fitting_and_testing_for_groups(
        run_dir, cur_iter, groups, Params(params.get('MD_bulk_defect_step'), cur_iter), Zs, 'last_op__md_energy', select_by_desc_method,
        descriptor_strs, params.get('global/config_selection_descriptor_local', default=False), rng=ctx.obj['rng'], verbose=verbose)

    _ = evaluate_iter_and_fit_all(cur_iter, run_dir, params, Params(params.get('MD_bulk_defect_step'), cur_iter),
                                  [grp['cur_confs'] for grp in groups.values()],
                                  [grp['testing_confs'] for grp in groups.values()],
                                  params.get('fit/database_modify_mod'),
                                  params.get('fit/calc_fitting_error', default=True),
                                  extra_fitting_files=params.get('fit/extra_fitting_files', default=[]),
                                  seeds=ctx.obj['seeds'], verbose=verbose)

    print_log('done')
    increment_active_iter(cur_iter)


@cli.command('reevaluate_and_fit_step')
@click.option('--cur_iter', type=click.INT, default=None)
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def do_reevaluate_and_fit_step(ctx, cur_iter, verbose):
    verbose = verbose or ctx.obj['verbose']
    cur_iter = process_cur_iter(ctx.obj['cur_iter'] if cur_iter is None else cur_iter)
    print_log(f'GAP_RSS_ITER_FIT DOING STEP reevaluate_and_fit_step {cur_iter}', blank_lines=True)

    params = ctx.obj['params']
    params.cur_iter = cur_iter
    run_dir, Zs, compositions = step_startup(params, cur_iter)

    fitting_configs = []
    for glob_i, old_fitting_glob in enumerate(params.get('reevaluate_and_fit_step/fitting_files')):

        old_fitting_files = []
        for old_file in glob.glob(old_fitting_glob):
            assert Path(old_file).is_file()
            old_fitting_files.append(old_file)

        print_log(f'Reevaluating and fitting to configs in existing files {old_fitting_files}')

        # no rundir, assuming that old_fitting_files are all relative to directory from which top level script is started
        reeval_configs_in = ConfigSet(old_fitting_files)
        reeval_configs_out = OutputSpec(f'DFT_evaluated_fitting.reevaluated_extra_glob_{glob_i}.xyz', file_root=run_dir)

        fitting_configs.append(evaluate_ref(reeval_configs_in, reeval_configs_out, params, run_dir, verbose))

    testing_configs = []
    for glob_i, old_testing_glob in enumerate(params.get('reevaluate_and_fit_step/testing_files')):

        old_testing_files = []
        for old_file in glob.glob(old_testing_glob):
            assert Path(old_file).is_file()
            old_testing_files.append(old_file)

        print_log(f'Reevaluating testing configs in existing files {old_testing_files}')

        # no rundir, assuming that old_testing_files are all relative to directory from which top level script is started
        reeval_configs_in = ConfigSet(old_testing_files)
        reeval_configs_out = OutputSpec(f'DFT_evaluated_testing.reevaluated_extra_glob_{glob_i}.xyz', file_root=run_dir)

        testing_configs.append(evaluate_ref(reeval_configs_in, reeval_configs_out, params, run_dir, verbose))

    GAP_xml_file = do_fit_and_test(cur_iter, run_dir, params, fitting_configs, None,
                                   database_modify_mod=params.get('fit/database_modify_mod'),
                                   seeds=ctx.obj['seeds'], verbose=verbose)

    return GAP_xml_file


def RSS_minima_diverse(run_dir, groups, step_params, Zs,
                       select_by_desc_method, config_selection_descriptor_strs, config_selection_descriptor_local,
                       prev_GAP, select_convex_hull, get_entire_trajectories, optimize_kwargs={}, rng=None, verbose=False):
    """do RSS, select diverse minima using flat histogram + descriptor-based, optionally convex hull

    Parameters
    ----------
    run_dir: pathlike
        run directory
    groups: dict
        groups to separate runs into
    step_params: Params
        run parameters
    Zs: list(int)
        all atomic numbers in system
    select_by_desc_method: str
        method for select by descriptor
    config_selected_descriptor_strs: list(str) / dict(Z : str)
        descriptors strings for by-descriptor selection
    config_selected_descriptor_local: bool
        selection descriptor is local
    prev_GAP: str
        full path for GAP file for RSS
    select_convex_hull: bool
        always select minima on (x, V, E) convex hull
    get_entire_trajectories: bool
        return entire RSS trajectories leading up to minima
    optimize_kwargs: dict, default {}
        optional kwargs for optimize call
    verbose: bool
        verbose output

    Returns
    -------
    None, but groups[grp_label]['cur_confs'] is set to minima or RSS traj configs (and optionally convex hull),
    and groups[grp_label]['convex_hull'] is set to just convex hull (or None)
    """

    for grp_label in groups:
        print_log(f'minimizing with {prev_GAP}')
        # do minims
        # ID preconditioner needed to make it not hang with multiprocessing - need to investigate why default, 'auto',
        # which should always result in None, hangs.  Could be weird volume jumps related to symmetrization cause
        # preconditioner neighbor list to go crazy.
        trajs = optimize.optimize(groups[grp_label]['cur_confs'],
                          OutputSpec(f'optimize_traj.{grp_label}.xyz', file_root=run_dir),
                          calculator=(Potential, None, {'param_filename': prev_GAP}),
                          precon='ID', keep_symmetry=True, rng=rng, **optimize_kwargs)

        print_log('selecting minima from trajectories')
        # select minima from trajs
        minima = wfl.select.simple.by_bool_func(trajs, OutputSpec(f'minima.{grp_label}.xyz', file_root=run_dir),
                                                lambda at: at.info["optimize_config_type"].startswith("optimize_last"))

        if select_convex_hull:
            print_log('selecting convex hull of minima')
            groups[grp_label]['convex_hull'] = wfl.select.convex_hull.select(
                minima,
                OutputSpec(f'minima_convex_hull.{grp_label}.xyz', file_root=run_dir),
                info_field='last_op__optimize_energy')
        else:
            groups[grp_label]['convex_hull'] = None

        exclude_list = groups[grp_label]['convex_hull']

        grp_frac = groups[grp_label]['frac']
        minima_flat_histo_kT = step_params.get('minima_flat_histo_kT', step_params.get('flat_histo_kT'))
        minima_config_by_desc, _ = flat_histo_then_by_desc(
            run_dir, minima, 'minima', grp_label, Zs, 'last_op__optimize_energy', minima_flat_histo_kT,
            int(step_params.get('minima_flat_histo_N') * grp_frac), select_by_desc_method,
            config_selection_descriptor_strs, config_selection_descriptor_local,
            int(step_params.get('minima_by_desc_select_N') * grp_frac), testing_N=0,
            vol_range=step_params.get('vol_range', 0.25), compos_range=step_params.get('composition_range', 0.01),
            by_desc_exclude_list=exclude_list, flat_histo_by_bin=step_params.get('flat_histo_by_bin', True),
            flat_histo_replacement=step_params.get('flat_histo_with_replacement', False),
            rng=rng, verbose=verbose)

        if select_convex_hull:
            selected_minima = ConfigSet([minima_config_by_desc, groups[grp_label]['convex_hull']])
        else:
            selected_minima = minima_config_by_desc

        if not get_entire_trajectories:
            groups[grp_label]['cur_confs'] = selected_minima
        else:
            print_log('selecting entire trajectories of selected minima')
            # find indices of selected minima
            selected_minima_config_i = set([at.info['buildcell_config_i'] for at in selected_minima])

            # select all configs for these indices from all trajs
            selected_traj = wfl.select.simple.by_bool_func(
                trajs, OutputSpec(f'selected_rss_traj.{grp_label}.xyz', file_root=run_dir),
                lambda at: at.info["buildcell_config_i"] in selected_minima_config_i)

            groups[grp_label]['cur_confs'] = selected_traj


def flat_histo_then_by_desc(run_dir, configs, file_label, grp_label, Zs,
                            E_info_field, flat_histo_kT, flat_histo_N, select_by_desc_method,
                            config_selection_descriptor_strs, config_selection_descriptor_local, by_desc_select_N,
                            testing_N, by_desc_exclude_list, vol_range=0.25, compos_range=0.01, prev_selected_descs=None,
                            flat_histo_by_bin=True, flat_histo_replacement=False,
                            rng=None, verbose=False):
    """select by doing flat histo (optionally) and then by descriptor

    Parameters
    ----------
    run_dir: str
        run directory
    configs: ConfigSet
        set of configs to pick from
    file_label: str
        label for files (that gets suffixes like _flat_histo and _by_desc appended to)
    grp_label: str
        group label
    Zs: list(int)
        list of all atomic numbers in system
    E_info_field: str
        atoms.info field to use for energy flat histogram
    flat_histo_kT: float
        kT to bias flat histo by, ignored if flat_histo_N is None
    flat_histo_N: int / None
        number of configs to pick with flat histo, None to not do flat histo
    select_by_desc_method: 'CUR', 'greedy_fps', 'greedy_fps_all_iters'
        method for selecting by config
    config_selection_descriptor_strs: list(str) / dict(str)
        descriptors strings to do selection by
    config_selection_descriptor_local: bool
        descriptors are per-atom
    by_desc_select_N: int
        number of configs to pick by descriptor
    testing_N: int
        number of extra testing configs to pick by descriptor
    by_desc_exclude_list: list(Atoms)
        configurations to exclude from by-descriptor selection
    vol_range: float, default 0.25
        range of vol/atom to be considered "nearby" when computing flat histogram energy distances
    compos_range: float, default 0.01
        range of composition x to be considered "nearby" when computing flat histogram energy distances
    flat_histo_by_bin: bool, default True
        do flat histogram selection by bin
    flat_histo_replacement: bool, default False
        do flat histogram selection with replacement
    prev_selected_descs: ndarray(n_descs, desc_len) / None, default None
        array of descriptors (row vectors) of previously selected configs
    verbose: bool, default False
        verbose output

    Returns
    -------
    configs_by_desc, testing_configs: configurations selected by descriptor
    """
    if config_selection_descriptor_local:
        raise RuntimeError('Selection by descriptor not implemented for local descriptors')

    if flat_histo_N is None or flat_histo_N == 0:
        configs_init = configs
    elif flat_histo_N > 0:
        print_log(f'computing energy relative to nearby configs for group {grp_label} file {file_label}')
        # select from configs with flat histogram in energy relative to "nearby" configs
        # NOTE: may eventually need to deal with extxyz read not storing energy in Atoms.info
        configs_rel_E = val_relative_to_nearby_composition_volume_min(
            configs, OutputSpec(f'{file_label}_E_rel_nearby.{grp_label}.xyz', file_root=run_dir),
            vol_range=vol_range, compos_range=compos_range,
            info_field_in=E_info_field,
            info_field_out='E_per_atom_dist_to_nearby',
            Zs=Zs, per_atom=True)

        print_log(f'selecting from configs with flat histogram, kT {flat_histo_kT}, for {file_label}')
        configs_init = biased_select_conf(configs_rel_E, OutputSpec(f'{file_label}_flat_histo.{grp_label}.xyz',
                                                                    file_root=run_dir),
                                          num=flat_histo_N, info_field='E_per_atom_dist_to_nearby',
                                          rng=rng, kT=flat_histo_kT,
                                          by_bin=flat_histo_by_bin,       ## Default changed, not compatible with old behavior
                                          replace=flat_histo_replacement,
                                          verbose=verbose)
    else:
        # flat_histo_N < 0 means select at random (UGLY HACK)
        # NOTE: the following should probably be refactored into a simple routine
        n_configs = sum([1 for at in configs])
        selected_inds = rng.choice(n_configs, size=-flat_histo_N, replace=False)
        configs_init = wfl.select.simple.by_index(configs,
            OutputSpec(f'{file_label}_random_init.{grp_label}.xyz', file_root=run_dir),
            selected_inds)

    if select_by_desc_method == 'random':
        n_configs = sum([1 for at in configs_init])
        selected_inds = rng.choice(n_configs, size=by_desc_select_N, replace=False)
        configs_selected = wfl.select.simple.by_index(configs_init,
            OutputSpec(f'{file_label}_random_selected.{grp_label}.xyz', file_root=run_dir),
            selected_inds)
        if testing_N > 0:
            avail_inds = set(list(range(n_configs)))
            avail_inds -= set(selected_inds)
            selected_testing_inds = rng.choice(list(avail_inds), size=testing_N, replace=False)
            testing_configs = wfl.select.simple.by_index(configs_init,
                OutputSpec(f'{file_label}_testing.{grp_label}.xyz', file_root=run_dir),
                selected_testing_inds)

    else:
        print_log(
            f'computing descriptors and selecting from (optionally) flat histogram by descriptor for {file_label} ' + str(
                config_selection_descriptor_strs))
        # calc descriptors and by-desc select from flat histo selected
        configs_flat_histo_with_desc = wfl.descriptors.quippy.calculate(
            configs_init, OutputSpec(f'{file_label}_with_desc.{grp_label}.xyz', file_root=run_dir),
            config_selection_descriptor_strs, 'config_selection_desc',
            per_atom=config_selection_descriptor_local,
            verbose=verbose)

        # no kwargs as default
        extra_kwargs = {}
        if select_by_desc_method == 'CUR':
            selector_func = wfl.select.by_descriptor.CUR_conf_global
            extra_kwargs['kernel_exp'] = 4 # fixme parameter
            extra_kwargs['rng'] = rng
        elif select_by_desc_method == 'greedy_fps' or select_by_desc_method == 'greedy_fps_all_iters':
            selector_func = wfl.select.by_descriptor.greedy_fps_conf_global
            extra_kwargs['rng'] = rng
            if select_by_desc_method == 'greedy_fps_all_iters':
                extra_kwargs['prev_selected_descs'] = prev_selected_descs
        else:
            raise RuntimeError(f'Unknown method for selection by descriptor "{select_by_desc_method}"')

        configs_selected = selector_func(configs_flat_histo_with_desc,
                                        OutputSpec(f'{file_label}_by_desc.{grp_label}.xyz', file_root=run_dir),
                                        num=by_desc_select_N, at_descs_info_key='config_selection_desc',
                                        keep_descriptor_info=False, exclude_list=by_desc_exclude_list, **extra_kwargs)
        if testing_N > 0:
            by_desc_exclude_list = ConfigSet([by_desc_exclude_list, configs_selected])
            testing_configs = selector_func(configs_flat_histo_with_desc,
                                            OutputSpec(f'{file_label}_testing.{grp_label}.xyz', file_root=run_dir),
                                            num=testing_N, at_descs_info_key='config_selection_desc',
                                            keep_descriptor_info=False, exclude_list=by_desc_exclude_list, **extra_kwargs)
        else:
            testing_configs = None

    return configs_selected, testing_configs


def calc_descriptors_to_file(run_dir, basename, grp_label, configs, descriptor_strs, descriptor_local, verbose=False):
    if descriptor_local:
        raise RuntimeError('calc_descriptors_to_file not yet supported for local descriptors')

    # skip if result file exists
    if os.path.exists(os.path.join(run_dir, f'{basename}.{grp_label}.average_desc.txt')):
        return

    configs_with_descs = wfl.descriptors.quippy.calculate(configs, OutputSpec(),
                                                  descriptor_strs, 'config_selection_desc', per_atom=descriptor_local,
                                                  verbose=verbose)

    np.savetxt(os.path.join(run_dir, f'tmp.{basename}.{grp_label}.average_desc.txt'),
               [at.info['config_selection_desc'] for at in configs_with_descs])

    os.rename(os.path.join(run_dir, f'tmp.{basename}.{grp_label}.average_desc.txt'),
              os.path.join(run_dir, f'{basename}.{grp_label}.average_desc.txt'))


def load_old_descriptors_arrays(run_dirs, basename, grp_label):
    if len(run_dirs) == 0:
        return None

    descriptors_array = []
    for run_dir in run_dirs:
        try:
            descriptors_array.append(np.loadtxt(os.path.join(run_dir, f'{basename}.{grp_label}.average_desc.txt')))
        except OSError:
            # Ignore missing files. Various innocuous causes, e.g. changing of groups between
            # iterations.
            pass

    if len(descriptors_array) > 0:
        return np.vstack(descriptors_array)
    else:
        return np.asarray([])


def select_fitting_and_testing_for_groups(run_dir, cur_iter, groups, step_params, Zs, E_info_field, select_by_desc_method,
                                          descriptor_strs, descriptor_local, flat_histo=True, rng=None, verbose=False):
    """select fitting and testing configurations from a pool of configs for each group

    Parameters
    ----------
    run_dir: pathlike
        run directory
    cur_iter: int
        current iteration # (for reading of previous iter selected config descriptors)
    groups: dict
        dict of groups to select separately, with keys for each group label,
        and within that 'cur_confs' key with pool of configs and 'convex_hull' for
        convex hull configs to exclude
    step_params: Params
        parameters for this kind of step
    Zs: list(int)
        all atomic numbers in system
    E_info_field: str
        atoms.info field to use for energy flat histogram
    select_by_desc_method: 'CUR' / 'greedy_fps' / 'greedy_fps_all_iters'
        method for selection by descriptor
    descriptor_strs: list(str) or dict(Z : str)
        descriptors for selection by descriptor
    descriptor_local: bool
        descriptor is per-atom
    flat_histo: bool, default True
        first do flat histogram
    verbose: bool, default False
        verbose output

    Returns
    -------
    None, but groups['cur_confs'] is selected fitting configs, and groups['testing_confs'[] is optional
        selected fitting configs.  Selected fitting descriptors are also saved, for selection by descriptor
        that uses info about all previous iters (e.g. greedy_fps_all_iters).
    """
    for grp_label in groups:
        grp_frac = groups[grp_label]['frac']

        # load previous descriptors
        prev_selected_descs = load_old_descriptors_arrays([f'run_iter_{i}' for i in range(cur_iter)],
                                                          'fitting_configs_descriptors', grp_label)

        # params for optional initial filter by flat histo
        if flat_histo:
            flat_histo_kT = step_params.get('flat_histo_kT')
            flat_histo_N = int(step_params.get('final_flat_histo_N') * grp_frac)
        else:
            flat_histo_kT = None
            flat_histo_N = None

        # do selection by flat histo (optional) and by descriptor
        fitting_confs, testing_confs = flat_histo_then_by_desc(
            run_dir, groups[grp_label]['cur_confs'], 'selected_fitting', grp_label, Zs, E_info_field, flat_histo_kT, flat_histo_N,
            select_by_desc_method, descriptor_strs, descriptor_local,
            int(step_params.get('fitting_by_desc_select_N') * grp_frac),
            testing_N=int(step_params.get('testing_by_desc_select_N') * grp_frac),
            by_desc_exclude_list=groups[grp_label].get('convex_hull', None), prev_selected_descs=prev_selected_descs,
            rng=rng, verbose=verbose)

        # save configs
        groups[grp_label]['cur_confs'] = fitting_confs
        groups[grp_label]['testing_confs'] = testing_confs

        # save newly selected descriptors (only cur_confs, not testing)
        # could save descs directly from flat_histo_then_by_desc (or by selection routine called in there),
        # but that would require digging in, and hopefully this recomputation is only a minor expense
        calc_descriptors_to_file(run_dir, 'fitting_configs_descriptors', grp_label, groups[grp_label]['cur_confs'],
                                 descriptor_strs, descriptor_local, verbose=verbose)


def step_startup(params, cur_iter):
    run_dir = f'run_iter_{cur_iter}'
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

    Zs = np.array(params.get('global/Zs'))
    compositions = params.get('global/compositions')

    return run_dir, Zs, compositions


if __name__ == '__main__':
    cli()
