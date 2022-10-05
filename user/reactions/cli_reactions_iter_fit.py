import json
import os
import pathlib
import pprint
import subprocess
import sys
from glob import glob
from ..generate.user.generate import collision

import click
import numpy as np
import yaml
try:
    from quippy.potential import Potential
except ModuleNotFoundError:
    pass


from wfl.calculators import committee
from wfl.calculators.orca import basinhopping 
from wfl.configset import ConfigSet, OutputSpec
from wfl.descriptor_heuristics import descriptors_from_length_scales
from wfl.generate.vib import sample_normal_modes
from wfl.select.simple import by_energy
from wfl.utils.logging import increment_active_iter, print_log, process_active_iter
from wfl.utils.params import Params
from wfl.utils.vol_composition_space import composition_space_Zs
from wfl.fit import gap
from ..generate import atoms_and_dimers
from ..reactions_processing import trajectory_processing
from ..select import weighted_cur
from wfl.utils.version import get_wfl_version


@click.group()
@click.option('--verbose', '-v', is_flag=True)
@click.option('--configuration', '-c', type=click.STRING, required=True)
@click.option('--active-iter', '-i', type=click.INT, default=None)
@click.pass_context
def cli(ctx, verbose, configuration, active_iter):
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['active_iter'] = active_iter

    print_log('\nGAP_REACTIONS_ITER_FIT STARTING, code version ' + get_wfl_version())

    with open(configuration, "r") as file:
        config = json.load(file)

    fragments_file = config["global"]["fragments_file"]
    config['global']['fragments'] = ConfigSet(input_files=fragments_file)

    # gather all chemical species across fragments
    config['global']['atomic_numbers'] = composition_space_Zs(config['global']['fragments'])

    if verbose:
        print('fragments', " ".join([at.get_chemical_formula() for at in config['global']['fragments']]))

    ctx.obj['params'] = Params(config)


@cli.command('prep')
@click.pass_context
def prep(ctx):
    print_log('\nGAP_REACTIONS_ITER_FIT DOING STEP prep')
    verbose = ctx.obj['verbose']
    params = ctx.obj['params']

    # elements and compositions
    atomic_numbers = np.array(params.get('global/atomic_numbers'))
    print_log('Zs ' + pprint.pformat(atomic_numbers, indent=2))

    # read length scales
    with open(params.get('prep/length_scales_file')) as fin:
        length_scales = yaml.safe_load(fin)

    # prep GAP fitting config using Zs, length scales
    fit_json = gap.multistage.prep_input(atomic_numbers, length_scales,
                                         params.get('fit/GAP_template_file'),
                                         sharpness=params.get('fit/universal_SOAP_sharpness', default=0.5))
    yaml.dump(fit_json, open('multistage_GAP_fit_settings.yaml', 'w'), indent=4)

    # similarly prep config-selection descriptor using Zs, length scales
    config_select_descriptor = params.get('global/config_selection_descriptor')
    config_selection_descs, desc_atomic_numbers = descriptors_from_length_scales(
        config_select_descriptor, atomic_numbers, length_scales)
    descs_by_z = {}
    for at_num in atomic_numbers:
        descs_by_z[int(at_num)] = [desc for zz, desc in zip(desc_atomic_numbers, config_selection_descs) if
                                   zz == at_num]
    yaml.dump(descs_by_z, stream=open('gap_reactions_iter_fit.prep.config_selection_descriptors.yaml', 'w'),
              default_flow_style=False)


@cli.command('initial-step')
@click.option('--active-iter', '-i', type=click.INT, default=None)
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def do_initial_step(ctx, active_iter, verbose):
    verbose = verbose or ctx.obj['verbose']
    active_iter = process_active_iter(ctx.obj['active_iter'] if active_iter is None else active_iter)
    print_log(f'\nGAP_REACTIONS_ITER_FIT DOING STEP initial-step {active_iter}')
    params = ctx.obj['params']
    atomic_numbers = params.get('global/atomic_numbers')
    fit_params = load_fit_params()
    params.cur_iter = active_iter
    run_dir = step_startup(params, active_iter)

    # prepare dimer XYZ
    dimers = OutputSpec(output_files='dimers.xyz', file_root=run_dir, force=True)
    atoms_and_dimers.prepare(
        dimers, atomic_numbers=atomic_numbers, dimer_n_steps=params.get("initial_step/dimer/n_steps"),
        dimer_factor_range=(params.get("initial_step/dimer/r_min"), params.get("initial_step/dimer/cutoff")),
        do_isolated_atoms=False, fixed_cell=[params.get("initial_step/dimer/cutoff") * 3] * 3)

    # DFT on dimers -- with masking the large forces if needed
    print_log('evaluating with DFT - dimers')
    dimers_dft_intermediate = OutputSpec(file_root=run_dir, output_files='dimers_intermediate.xyz',
                                            all_or_none=True, force=True)
    dft_dimers = evaluate_dft(dimers.to_ConfigSet(), dimers_dft_intermediate, params, run_dir)

    # create glue & e0
    print_log('write baseline GLUE model')
    glue_param_str, e0_dict = gap.glue_2b.construct_glue_2b(dft_dimers, "REF_energy",
                                                            cutoff=params.get("initial_step/dimer/cutoff"),
                                                            filename=fit_params.get('core_ip_file'))

    # write some of the dimers to file for training
    print_log('write dimers to file for training')
    dimers_out = OutputSpec(file_root=run_dir, output_files='DFT_evaluated.dimers.xyz',
                               all_or_none=True, force=True)
    _ = by_energy(dft_dimers, dimers_out, e0=e0_dict, lower_limit=None,
                  upper_limit=params.get("initial_step/dimer/inclusion_energy_upper_limit", 0.))

    # write e0 to file
    print_log('writing isolated atoms')
    isolated_atoms = OutputSpec(file_root=run_dir, output_files='DFT_evaluated.isolated_atoms.xyz', all_or_none=True,
                                 force=True)
    atoms_and_dimers.isolated_atom_from_e0(isolated_atoms, e0_dict, cell_size=2 * params.get("initial_step/dimer/cutoff"),
                                           energy_key="REF_energy")

    # normal mode sampling with glue
    print_log('normal mode sampling started')
    normal_modes_in = OutputSpec(file_root=run_dir, output_files="normal_modes.xyz",
                                    all_or_none=True, force=True)
    sample_normal_modes(frames=params.get("global/fragments"),
                        output=normal_modes_in,
                        calculator=(Potential, None, dict(args_str="IP Glue", param_str=glue_param_str)),
                        nfree=params.get("initial_step/NormalModes/n_free", 2),
                        prefix="vib_", config_type="normal_modes",
                        num_per_mode=params.get("initial_step/NormalModes/num_per_mode"))

    # DFT on fragments
    print_log('evaluating with DFT - fragments')
    fragments_dft_out = OutputSpec(file_root=run_dir, output_files='DFT_evaluated.fragments.xyz',
                                      all_or_none=True, force=True)
    # fixme: add config_type here to the file directly somehow
    dft_fragments = evaluate_dft(
        ConfigSet(input_configs=[at for at in params.get("global/fragments") if len(at) > 2]),
        fragments_dft_out, params, run_dir)

    # DFT on normal mode data
    print_log('evaluating with DFT - normal modes')
    normal_modes_dft_out = OutputSpec(file_root=run_dir, output_files='DFT_evaluated.normal_modes.xyz',
                                         all_or_none=True, force=True)
    dft_normal_modes = evaluate_dft(normal_modes_in.to_ConfigSet(), normal_modes_dft_out, params, run_dir)

    # fit
    print_log('fitting')
    database_configs = ConfigSet(input_configsets=[dimers_out.to_ConfigSet(), dft_normal_modes,
                                                      isolated_atoms.to_ConfigSet(), dft_fragments])
    # WARNING: OUTDATED CALL - NEED TO UPDATE TO DO DATABASE MODIFY BEFORE AND REF ERROR CALC AFTER
    _ = gap.multistage.fit(database_configs, GAP_name='GAP_iter_0', params=fit_params,
                           database_modify_mod=params.get('fit/database_modify_mod'),
                           run_dir=run_dir, skip_if_present=True, verbose=verbose, ref_property_prefix="REF_",
                           num_committee=params.get("fit/num_committee"),
                           field_error_scale_factors=params.get("fit/field_error_scale_factors"),
                           save_err_configs_basename=f'error_database.GAP_iter_0.xyz')
    print_log('done')
    increment_active_iter(active_iter)


@cli.command('md-step')
@click.option('--active-iter', '-i', type=click.INT, default=None)
@click.option('--verbose', '-v', is_flag=True)
@click.option('--skip-collision', '-v', is_flag=True,
              help="Dev flag for skipping collision calculations, which cannot be skipped yet automatically")
@click.option("--do-neb", is_flag=True, help="Calculate NEB between differing relaxed frames")
@click.option("--do-ts-irc", is_flag=True, help="Calculate TS & IRC on NEBs")
@click.pass_context
def do_md_step(ctx, active_iter, verbose, skip_collision, do_neb, do_ts_irc):
    verbose = verbose or ctx.obj['verbose']
    active_iter = process_active_iter(ctx.obj['active_iter'] if active_iter is None else active_iter)
    print_log(f'\nGAP_REACTIONS_ITER_FIT DOING STEP md-step {active_iter}')
    params = ctx.obj['params']
    fit_params = load_fit_params()
    params.cur_iter = active_iter
    run_dir = step_startup(params, active_iter)
    descriptor_strs = yaml.load(open('gap_reactions_iter_fit.prep.config_selection_descriptors.yaml'),
                                Loader=yaml.FullLoader)
    if do_ts_irc and not do_neb:
        raise ValueError("Cannot perform TS calculation without NEB first")

    # 0. find previous GAP model
    prev_gap_committee = read_previous_gap_models(active_iter)
    prev_gap_main = prev_gap_committee[0]

    # 1. run collisions
    # fixme: this cannot be skipped if done, because of the tempdir-like directories
    if not skip_collision:
        for velocity_params in params.get("collision_step/kwargs/velocity_params"):
            # this is non-ideal but we need to change the collision runner to support multiple velocity params passed
            # and parallel run over them as well, rather than blocks of them as now
            collision_kw = dict(params.get("collision_step/kwargs"), velocity_params=velocity_params)
            collision.multi_run_all_with_all(fragments=[at for at in params.get("global/fragments")],
                                             param_filename=prev_gap_main,
                                             workdir=run_dir,
                                             n_pool=int(os.environ.get('WFL_AUTOPARA_NPOOL', 1)),
                                             **collision_kw)

    # 2. calculate energies with committee of models
    collision_with_committee_results = calc_gap_committee("collision_*/collision.raw_md.xyz",
                                                          prev_gap_committee, run_dir)

    print("collision outputs with committee:", collision_with_committee_results, "\n")

    # 3. selection with weighted-CUR from the MD data
    # wfl -v select-configs weighted-cur -f --limit=0.10 --cut-threshold 0.85 -o selected.xyz --stride=4
    # -n 1 50 -n 6 50 -n 8 50 md/collision_*/committee.collision.raw_md.xyz
    selected_out = OutputSpec(output_files="selected.weighted_CUR.xyz", force=True, all_or_none=True,
                                 file_root=run_dir)
    num_select = {key: val for key, val in params.get("collision_step/selection/num_select")}
    weighted_cur.selection_full_desc(collision_with_committee_results, selected_out,
                                     descriptor_strs=descriptor_strs, n_select=num_select,
                                     limit=params.get("collision_step/selection/lower_energy_limit"))

    # 4.0 NEB+TS calculation for all collisions
    if do_neb:
        # todo: enable skipping this if done
        seeds = [trajectory_processing.xyz_to_seed(fn) for fn in
                 sorted(glob(os.path.join(os.path.abspath(run_dir), "collision_*/collision.raw_md.xyz")))]

        collision.post_process_collision(
            seed=seeds,
            calc=(Potential, None, {'param_filename': prev_gap_main}),
            do_neb=do_neb, do_ts_irc=do_ts_irc,
            minim_kwargs=params.get("neb_step/minim_kwargs"),
            minim_interval=params.get("neb_step/minim_interval", 50),
            neb_kwargs=params.get("neb_step/neb_kwargs"),
            ts_kwargs=params.get("neb_step/ts_kwargs"),
            irc_kwargs=params.get("neb_step/irc_kwargs"),
            n_pool=int(os.environ.get('WFL_AUTOPARA_NPOOL', 1)),
        )

        # 4.1 calculate committee on NEB and TS as well
        neb_select_from = calc_gap_committee("collision_*/collision.neb_frames.xyz",
                                             prev_gap_committee, run_dir)
        if do_ts_irc:
            ts_committee = calc_gap_committee("collision_*/collision.neb_ts.xyz", prev_gap_committee, run_dir)
            neb_select_from.merge(ts_committee)

        # 4.2 selection from NEB + TS
        selected_neb_out = OutputSpec(output_files="selected.weighted_CUR.neb_and_ts.xyz", force=True,
                                         all_or_none=True,
                                         file_root=run_dir)
        num_select_neb = {key: val for key, val in params.get("neb_step/selection/num_select")}
        weighted_cur.selection_full_desc(neb_select_from, selected_neb_out,
                                         descriptor_strs=descriptor_strs, n_select=num_select_neb,
                                         limit=params.get("neb_step/selection/lower_energy_limit"))

        # 4.3 DFT on selected NEB frames
        print_log('evaluating with DFT - NEB/TS samples')
        dft_out_neb = OutputSpec(file_root=run_dir, output_files='DFT_evaluated.NEB-selected.xyz',
                                    all_or_none=True, force=True)
        _ = evaluate_dft(selected_neb_out.to_ConfigSet(), dft_out_neb, params, run_dir)
    else:
        dft_out_neb = None

    # 5. DFT calculation on selected frames
    print_log('evaluating with DFT - MD samples')
    dft_out = OutputSpec(file_root=run_dir, output_files='DFT_evaluated.MD-selected.xyz',
                            all_or_none=True, force=True)
    # fixme: add config_type here to the file directly somehow
    _ = evaluate_dft(selected_out.to_ConfigSet(), dft_out, params, run_dir)

    # fit
    print_log('fitting')
    old_configs = [os.path.join('run_iter_{}'.format(i), 'DFT_evaluated.*.xyz') for i in range(0, active_iter)]
    old_dft_evaluated_configs = ConfigSet(input_files=old_configs)
    database_configs = ConfigSet(input_configsets=[old_dft_evaluated_configs, dft_out.to_ConfigSet()])
    if dft_out_neb is not None:
        database_configs.merge(dft_out_neb.to_ConfigSet())
    print_log("fitting database is: " + str(database_configs) + "\n")
    # WARNING: OUTDATED CALL - NEED TO UPDATE TO DO DATABASE MODIFY BEFORE AND REF ERROR CALC AFTER
    _ = gap.multistage.fit(database_configs, GAP_name='GAP_iter_{}'.format(active_iter),
                           params=fit_params, database_modify_mod=params.get('fit/database_modify_mod'),
                           run_dir=run_dir, skip_if_present=True, verbose=verbose, ref_property_prefix="REF_",
                           num_committee=params.get("fit/num_committee"),
                           field_error_scale_factors=params.get("fit/field_error_scale_factors"),
                           save_err_configs_basename=f'error_database.GAP_iter_{active_iter}.xyz')

    print_log('done')
    increment_active_iter(active_iter)


def step_startup(params, active_iter):
    run_dir = f'run_iter_{active_iter}'
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

    return run_dir


def evaluate_dft(dft_in_configs, dft_evaluated_configs, params, run_dir):
    """ DFT evaluation, parameters taken from

    Parameters
    ----------
    dft_in_configs : ConfigSet
    dft_evaluated_configs : OutputSpec
    params : Params
        run parameters

    Returns
    -------
    evaluated_configs : ConfigSet
        as got from iterable_loop of the evaluators

    """

    # todo: merge this with the RSS one
    if params.dft_code == "ORCA":

        if dft_evaluated_configs.is_done():
            sys.stderr.write(
                'Returning before ORCA calculation since output is done on configset:\n' + str(
                    dft_evaluated_configs))
            return dft_evaluated_configs.to_ConfigSet()

        # only non-periodic solution possible
        return basinhopping.evaluate_basin_hopping(inputs=dft_in_configs,
                                         outputs=dft_evaluated_configs,
                                         workdir_root=run_dir,
                                         orca_kwargs=params.dft_params.get("kwargs", {}),
                                         output_prefix='REF_')
    else:
        raise NotImplementedError("DFT code not implemented for reactions:", params.dft_code)


def load_fit_params(filename='multistage_GAP_fit_settings.yaml'):
    """load fitting parameters, and check content

    Parameters
    ----------
    filename: path_like, default 'multistage_GAP_fit_settings.yaml'
        path to file to load YAML from

    Returns
    -------
    fit_params: dict
        loaded json file

    """
    with open(filename) as file:
        fit_params = yaml.safe_load(file)

    for p in ["core_ip_args", "core_ip_file"]:
        if p not in fit_params:
            raise RuntimeWarning(f"{p} not set in GAP settings, this may lead to issues with the MD "
                                 f"performance of your model for reactions")

    core_ip_args = str(fit_params.get("core_ip_args", "ip glue"))
    if core_ip_args.lower().strip() != "ip glue":
        raise RuntimeWarning("core_ip_args != 'ip glue' in GAP fit settings, which may be an issue unless "
                             "you have hacked in a different baseline model")

    return fit_params


def read_previous_gap_models(active_iter):
    """Read models

    Parameters
    ----------
    active_iter: int

    Returns
    -------
    main_model: str
        path to main gap model
    committee_of_models: list(str)
        full path to committee of models, includes main_model as well

    """
    previous_run_dir = os.path.abspath(f'run_iter_{active_iter - 1}')
    committee_of_models = sorted(
        glob(os.path.join(previous_run_dir, f'GAP_iter_{active_iter - 1}.committee_*.xml')))

    if len(committee_of_models) == 0:
        raise RuntimeError("Previous GAP models not found.")

    return committee_of_models


def calc_gap_committee(input_glob, gap_fn_list, run_dir, prefix="gap."):
    # GAP models
    gap_model_list = [(Potential, "", dict(param_filename=fn)) for fn in gap_fn_list]

    # process file names
    input_files = sorted(glob(os.path.join(os.path.abspath(run_dir), input_glob)))
    input_files = [f for f in input_files if os.path.getsize(f) > 0]  # omit empty files, Configset issue #70
    outputs = {fn: os.path.join(os.path.dirname(fn), f"{prefix}{os.path.basename(fn)}") for fn in input_files}

    # configsets -- with input file specified in
    configset = ConfigSet(input_files=input_files)
    outputspec = OutputSpec(output_files=outputs, force=True, all_or_none=True)

    # skip if done
    if outputspec.is_done():
        sys.stderr.write(
            'Returning before GAP-committee calculation since output is done on configset:\n' + str(outputspec))
        return outputspec.to_ConfigSet()

    # the calculation with all models
    outputspec.pre_write()
    for chunk in configset.group_iter():
        out_chunk = committee.calculate_committee(chunk, gap_model_list, output_prefix="gap_committee_{}_",
                                                  properties=['energy', 'forces'])
        outputspec.write(out_chunk, from_input_file=configset.get_current_input_file())
    outputspec.end_write()

    return outputspec.to_ConfigSet()


if __name__ == '__main__':
    cli(auto_envvar_prefix='GRIF')
