import sys
import warnings
from copy import deepcopy
from pathlib import Path
from xml.etree import cElementTree

import ase.io
import numpy as np
import yaml

from wfl.configset import ConfigSet
from wfl.descriptor_heuristics import descriptor_2brn_uniform_file, descriptors_from_length_scales
from wfl.fit.gap.simple import run_gap_fit
from wfl.utils.quip_cli_strings import dict_to_quip_str
from wfl.autoparallelize.utils import get_remote_info
from ..modify_database.scale_orig import modify as modify_scale_orig

try:
    from expyre import ExPyRe
except ModuleNotFoundError:
    warnings.warn('No expyre, so no remote runs')
    pass


# add stress to this when gap#7
GAP_fit_properties = ['energy', 'forces', 'virial', 'hessian', 'stress']

try:
    from quippy.descriptors import Descriptor
    from quippy.potential import Potential
except ModuleNotFoundError:
    warnings.warn('No quippy, so no multistage GAP fitting')
    pass

try:
    from universalSOAP import SOAP_hypers
except ModuleNotFoundError:
    warnings.warn('No universalSOAP, so no hyperparameters from universal SOAP heuristics (add_species="manual_universal_SOAP")')
    pass


def prep_params(Zs, length_scales, GAP_template, spacing=1.5,
                no_extra_inner=False, no_extra_outer=False, sharpness=1.0):
    """prepare parameters for multistage fitting based on YAML template file

    Parameters
    ----------
    Zs: list(int)
        atomic numbers
    length_scales: dict(Z: dict('bond_len': bond_len))
        dict with bond lengths for each atomic number Z
    GAP_template: dict or str
        dict with settings template (to get descriptors auto-filled from universal SOAP length scales) or filename with input template YAML
    """
    if isinstance(GAP_template, dict):
        multistep_gap_settings = GAP_template
    else:
        with open(GAP_template) as fin:
            multistep_gap_settings = yaml.safe_load(fin)

    try:
        hypers = SOAP_hypers(Zs, length_scales, spacing, no_extra_inner, no_extra_outer, sharpness=sharpness)
    except:
        hypers = None

    for (i_stage, stage) in enumerate(multistep_gap_settings['stages']):
        use_descs, _ = descriptors_from_length_scales(stage['descriptors'], Zs, length_scales,
                                                      SOAP_hypers=hypers)

        descriptor_2brn_uniform_file(use_descs, ident=f'_stage_{i_stage}')
        stage['descriptors'] = use_descs

    return multistep_gap_settings


def _select_info(ats, info_keys):
    for at in ats:
        for k in list(at.info.keys()):
            if k not in info_keys:
                del at.info[k]


# WARNING: this is hardwired to the names of fields in specific descriptors
# such as SOAP and turboSOAP
def max_cutoff(params):
    max_c = 0.0
    for stage in params['stages']:
        for desc in stage['descriptors']:
            try:
                max_c = max(max_c, desc['descriptor']['cutoff'])
            except KeyError:
                max_c = max(max_c, desc['descriptor']['rcut_hard'])
    return max_c


# noinspection PyPep8,PyPep8Naming
def fit(fitting_configs, GAP_name, params, ref_property_prefix='REF_',
        seeds=None, skip_if_present=False, run_dir='.',
        num_committee=0, committee_extra_seeds=None, committee_name_postfix='.committee_',
        verbose=False, remote_info=None, remote_label=None, wait_for_results=True):
    """Fit a GAP iteratively, setting delta from error relative to previous stage

    Parameters
    ----------
    fitting_configs: ConfigSet
        input fitting configurations
    GAP_name: str
        name of GAP label, also used as part of xml file
    params: dict
        parameters controlling each stage of fit, typically read in from YAML file
    ref_property_prefix: str, default 'REF\_'
        string prefix added to atoms.info/arrays keys (energy, forces, virial, stress, hessian)
    seeds: list(int)
        random seeds for each stage of fitting
    verbose: bool, default False
        verbose output
    skip_if_present: bool, default False
        skip if final GAP file exists in expected place
    run_dir: str, default '.'
        directory to run fitting in
    num_committee: int, default 0
        number of models to create as a committee of GAP models
        refits the last stage a total of this many times
    committee_extra_seeds: list(int) / None
        random seeds to use for committee of models after 0th
    committee_name_postfix: str, default ".committee_"
        str to add to name of committee models in the format: "{GAP_name}{committee_name_postfix}{num}.xml"
    remote_info: dict or wfl.autoparallelize.utils.RemoteInfo, or '_IGNORE' or None
        If present and not None and not '_IGNORE', RemoteInfo or dict with kwargs for RemoteInfo
        constructor which triggers running job in separately queued job on remote machine.  If None,
        will try to use env var WFL_EXPYRE_INFO used (see below). '_IGNORE' is for
        internal use, to ensure that remotely running job does not itself attempt to spawn another
        remotely running job.
    remote_label: str, default None
        label to match in WFL_EXPYRE_INFO
    wait_for_results: bool, default True
        wait for results of remotely executed job, otherwise return after starting job

    Environment Variables
    ---------------------
    WFL_EXPYRE_INFO: JSON dict or name of file containing JSON with kwargs for RemoteInfo
        contructor to be used to run fitting in separate queued job
    WFL_GAP_FIT_OMP_NUM_THREADS: number of threads to set for OpenMP of gap_fit

    Returns
    -------
    final_GAP_file, final_GAP_name
        string name of final GAP xml file, and name of corresponding GAP (xml label)
    """
    run_dir = Path(run_dir)

    if num_committee > 0:
        final_GAPnames = [f'{GAP_name}{committee_name_postfix}{i}' for i in range(num_committee)]
    else:
        final_GAPnames = [GAP_name]
    final_GAPfiles = [str(run_dir / (name + '.xml')) for name in final_GAPnames]

    if skip_if_present:
        try:
            # make sure all potentials are parseable
            for final_GAPfile in final_GAPfiles:
                # this should be redundant with the Potential() reading below, except that if the file
                # has a directory component QUIP's Fortran code will terminate rather than raising a
                # python exception.
                if not Path(final_GAPfile).is_file():
                    raise FileNotFoundError
                # NOTE: at least some FoX XML parsing errors lead to an immediate termination,
                # rather than a python exception that will actually be caught here.
                _ = Potential(param_filename=final_GAPfile)

            if num_committee > 0:
                return final_GAPfiles, final_GAPnames
            else:
                return final_GAPfiles[0], final_GAPnames[0]
        except (FileNotFoundError, RuntimeError):
            # Potential seems to return RuntimeError when file is missing
            pass

    if remote_info != '_IGNORE':
        remote_info = get_remote_info(remote_info, remote_label)

    if remote_info is not None and remote_info != '_IGNORE':
        input_files = remote_info.output_files.copy()
        output_files = remote_info.output_files.copy() + [str(run_dir)]

        fitting_configs = ConfigSet(list(fitting_configs))

        # set number of threads in queued job, only if user hasn't set them
        if not any([var.split('=')[0] == 'WFL_GAP_FIT_OMP_NUM_THREADS' for var in remote_info.env_vars]):
            remote_info.env_vars.append('WFL_GAP_FIT_OMP_NUM_THREADS=$EXPYRE_NUM_CORES_PER_NODE')
        if not any([var.split('=')[0] == 'WFL_NUM_PYTHON_SUBPROCESSES' for var in remote_info.env_vars]):
            remote_info.env_vars.append('WFL_NUM_PYTHON_SUBPROCESSES=$EXPYRE_NUM_CORES_PER_NODE')

        xpr = ExPyRe(name=remote_info.job_name, pre_run_commands=remote_info.pre_cmds, post_run_commands=remote_info.post_cmds,
                     env_vars=remote_info.env_vars, input_files=input_files, output_files=output_files, function=fit,
                     kwargs={'fitting_configs': fitting_configs, 'GAP_name': GAP_name, 'params': params,
                             'ref_property_prefix': ref_property_prefix,
                             'seeds': seeds, 'skip_if_present': skip_if_present, 'run_dir': run_dir,
                             'num_committee': num_committee, 'committee_extra_seeds': committee_extra_seeds,
                             'committee_name_postfix': committee_name_postfix, 'verbose': verbose, 'remote_info': '_IGNORE'})

        xpr.start(resources=remote_info.resources, system_name=remote_info.sys_name, header_extra=remote_info.header_extra,
                  exact_fit=remote_info.exact_fit, partial_node=remote_info.partial_node)

        if not wait_for_results:
            return None
        results, stdout, stderr = xpr.get_results(timeout=remote_info.timeout, check_interval=remote_info.check_interval)

        sys.stdout.write(stdout)
        sys.stderr.write(stderr)

        # no outputs to rename since everything should be in run_dir
        xpr.mark_processed()

        return results

    assert isinstance(ref_property_prefix, str) and len(ref_property_prefix) > 0

    if not run_dir.exists():
        run_dir.mkdir(parents=True)

    max_seed = np.iinfo(np.int32).max

    # read configs into a list in memory
    fitting_configs = ConfigSet(list(fitting_configs))

    # delete calculators so as to not confuse the issue
    for at in fitting_configs:
        at.calc = None

    # things to be added to the fitting line (in addition to strings in the params dict)
    fitting_line_kwargs = {}

    # create part of arg line with *_parameter_name arguments to pass to gap_fit
    for k in GAP_fit_properties:
        fitting_line_kwargs[f'{k.replace("forces", "force")}_parameter_name'] = ref_property_prefix + k

    # add core IP if needed
    if 'core_ip_args' in params:
        fitting_line_kwargs["core_ip_args"] = '{{{}}}'.format(params['core_ip_args'])
    if 'core_ip_file' in params:
        fitting_line_kwargs["core_param_file"] = params['core_ip_file']

    ref_energy_key = ref_property_prefix + 'energy'

    # gather set of all species to be fit
    Zs = set([Z for at in fitting_configs for Z in at.numbers])

    # gather e0
    e0s = {}
    for at in fitting_configs:
        if 'config_type' not in at.info:
            at.info['config_type'] = '_NONE_'
        if at.info['config_type'] == 'isolated_atom':
            Z = at.get_atomic_numbers()[0]
            if Z in e0s.keys():
                raise Exception('got more than one isolated_atom config for Z={}'.format(Z))
            else:
                if any(at.pbc):
                    raise RuntimeError('config_type==isolated_atom Z={} does not have all(pbc==False)'.format(Z))
                e0s[Z] = at.info[ref_energy_key] / len(at)
    if verbose:
        print('e0s', e0s)
    # check to make sure all Zs have e0 value set
    for Z in Zs:
        if Z not in e0s:
            raise RuntimeError('Did not find config_type==isolated_atom (to get e0) for Z {}'.format(Z))

    # create core Potential
    if 'core_ip_args' in params or 'core_ip_file' in params:
        core_pot = Potential(args_str=params.get('core_ip_args', None), param_filename=params.get('core_ip_file', None))
    else:
        core_pot = None

    # random seeds if none were provided
    rg = np.random.default_rng()
    if seeds is None:
        # use int32 for compatibility with fortran, which is what will use these values
        seeds = [int(rg.integers(max_seed, dtype=np.int32)) for _ in range(len(params['stages']))]
    if num_committee > 0 and committee_extra_seeds is None:
        committee_extra_seeds = [int(rg.integers(max_seed, dtype=np.int32)) for _ in range(num_committee - 1)]

    error_scale_factors = [stage.get('error_scale_factor', 1.0) for stage in params['stages']]

    # for now assume last descriptor is SOAP so delta is just energy error
    # NOTE: some related info is available from dimension of data structure returned by
    # Descriptor.calc, but that doesn't work for 2b where you want a different cutoff.
    # Maybe the dict that describes fitting stage should communicate this, and not just the
    # alternate cutoff for counting purposes.

    skipped_prev_iter = False
    descriptor_dicts = []
    prev_GAP = None
    for (i_stage, stage) in enumerate(params['stages']):
        if i_stage == len(params['stages']) - 1:
            # last stage, just give it final name
            GAPfile = final_GAPfiles[0]
        else:
            GAPfile = f'{run_dir / GAP_name}.stage_{i_stage}.xml'

        print('doing stage', i_stage, stage)

        if skip_if_present and Path(GAPfile).is_file():
            print(f'stage {i_stage} already done, skipping')
            skipped_prev_iter = True
            continue

        # read back from prev iter's file, in case in this run previous iter was skipped
        if i_stage > 0:
            prev_GAPfile = f'{run_dir / GAP_name}.stage_{i_stage - 1}.xml'
            prev_GAP = Potential(param_filename=prev_GAPfile)

            if skipped_prev_iter:
                with open(prev_GAPfile + '.descriptor_dicts.yaml') as fin:
                    descriptor_dicts = yaml.safe_load(fin)
                descriptor_dicts = descriptor_dicts[0:i_stage]

        ####################################################################################################
        # WORKAROUND FOR PY/FORTRAN EXTXYZ ISSUES
        # keep only info/arrays quantities that we actually need, to avoid Fortran choking on weird formatting
        _select_info(fitting_configs,
                     info_keys=[ref_property_prefix + k for k in GAP_fit_properties if k != "forces"] +
                               ['energy_sigma', 'force_sigma', 'virial_sigma', 'hessian_sigma'] +
                               ['_orig_energy_sigma', '_orig_force_sigma', '_orig_virial_sigma', '_orig_hessian_sigma'] +
                               ['config_type'])
        ####################################################################################################
        if any([f != 1.0 for f in error_scale_factors]):
            # modify database using this stage's error_scale_factor.
            # NOTE: somewhat ugly hack to reproduce previous behavior: exclude isolated_atom and dimer because
            # they were previously excluded by gap_rss_set_config_sigmas_from_convex_hull.py.modify(), which
            # is now being called manually from outside fitting function
            modify_scale_orig(fitting_configs, error_scale_factors[i_stage], config_type_exclude=['isolated_atom', 'dimer'])

        database_file = run_dir / f'fitting_database.combined.{GAP_name}.stage_{i_stage}.extxyz'
        ase.io.write(database_file, fitting_configs)
        database_ci = ConfigSet(database_file)

        # compute number of descriptors for i_stage'th one
        count_descs = []
        count_desc_cutoffs = []
        for desc_str_dict in stage['descriptors']:
            count_desc_str = dict_to_quip_str(desc_str_dict['descriptor'])
            if verbose:
                print('creating counting descriptor from string \'{}\''.format(count_desc_str))
            desc = Descriptor(count_desc_str)
            count_descs.append(desc)
            # keep track of short range for counting purposes if specified
            if 'count_cutoff' in desc_str_dict:
                use_cutoff = desc_str_dict['count_cutoff']
            else:
                use_cutoff = None
            count_desc_cutoffs.append(use_cutoff)

        descriptor_count = 0
        at_Ns = []
        for at in fitting_configs:
            # does it make sense to count descriptors based on what structures have _energy_ specified?
            if ref_energy_key not in at.info:
                continue

            for (d, c) in zip(count_descs, count_desc_cutoffs):
                descriptor_count += d.sizes(at, cutoff=c)[0]

            at_Ns.append(len(at))
        descriptor_count = float(descriptor_count)
        assert descriptor_count > 0
        if verbose:
            print('got descriptor_count {} per atom {}'.format(descriptor_count, descriptor_count / np.sum(at_Ns)))

        dEs = []
        for at in fitting_configs:
            if ref_energy_key in at.info:
                Eref = at.info[ref_energy_key]
                if i_stage == 0:
                    # compute variance of energy/atom (subtracting e0 and core if needed)
                    if core_pot is not None:
                        at.calc = core_pot
                        Eref -= at.get_potential_energy()
                    Eref -= np.sum([e0s[Z] for Z in at.get_atomic_numbers()])
                    dEs.append(Eref)
                else:
                    # compute energy/atom residual
                    Eref = at.info[ref_energy_key]
                    at.calc = prev_GAP
                    Egap = at.get_potential_energy()
                    dEs.append(Egap - Eref)
                    if verbose:
                        print('stage {} config {} residual error {} {}'.format(i_stage, at.info.get('config_type'),
                                                                               Egap / len(at), Eref / len(at)))

        abs_dE_sum = np.sum(np.abs(dEs))
        print('energy error MAE per atom {} per descriptor {}'.format(abs_dE_sum / np.sum(at_Ns),
                                                                      abs_dE_sum / descriptor_count))
        delta = abs_dE_sum / descriptor_count

        if 'delta_factors' in params:
            delta *= float(params['delta_factors'][i_stage])
        print('Got delta', delta)

        for stage_desc_dict in stage['descriptors']:
            # combine descriptor and fit dict contents
            descriptor_dict = deepcopy(stage_desc_dict['descriptor'])
            descriptor_dict.update(stage_desc_dict['fit'])
            # append delta as plain float for yaml safe_load
            descriptor_dict['delta'] = float(delta)
            # append add_species if specified (e.g. by heuristic duplicating code)
            if 'add_species' in stage_desc_dict:
                descriptor_dict['add_species'] = stage_desc_dict['add_species']

            descriptor_dicts.append(descriptor_dict)

        if verbose:
            print('descriptor_dicts', descriptor_dicts)

        # set input and output files for this fit
        fitting_line_kwargs["gap_file"] = GAPfile

        # add rng seed
        fitting_line_kwargs["rnd_seed"] = seeds[i_stage]

        # combine gap_params content from params dict with entries set by this routine in fitting_line_kwargs
        gap_simple_fit = deepcopy(params.get('gap_params', {}))
        gap_simple_fit['_gap'] = descriptor_dicts
        gap_simple_fit.update(fitting_line_kwargs)

        # save for next iter (if it might be restarted and this iter will be skipped)
        with open(GAPfile + '.descriptor_dicts.yaml', 'w') as fout:
            yaml.dump(descriptor_dicts, fout)
        # save for final committee fit
        with open(GAPfile + '.fitting_dicts.yaml', 'w') as fout:
            yaml.dump(gap_simple_fit, fout)

        # do a simple fit using gap_simple_fit
        stdout_file = run_dir / f'stdout.{GAP_name}.stage_{i_stage}.gap_fit'
        # If this function was called without a remote_info that applies to it, remote_info here
        # will still be the real remote_info, which can then be used in the underlying simple fits
        # If we're here with the multistage fit running remotely, the wrapper above passed
        # _IGNORE as remote_info, so these fits will run in this job, not their own separate remote
        # jobs.
        run_gap_fit(database_ci, gap_simple_fit, stdout_file=stdout_file, verbose=verbose,
                    remote_info=remote_info, remote_label=remote_label)

        print('')

    # rename final GAP
    GAP_xml_modify_label(final_GAPfiles[0], new_label=final_GAPnames[0])

    # read back, in case actual fitting was skipped
    with open(final_GAPfiles[0] + '.fitting_dicts.yaml') as fin:
        gap_simple_fit = yaml.safe_load(fin)

    # fitting more models for committee, 0 was already fit in regular multi-stage process
    for i_committee in range(1, num_committee):
        GAPfile = final_GAPfiles[i_committee]
        GAPname = final_GAPnames[i_committee]
        if skip_if_present and Path(GAPfile).is_file():
            print(f'committee {i_committee} already done, skipping')
            continue

        # set new output file
        gap_simple_fit["gap_file"] = GAPfile

        # set the random seed, this is making the resultant models different
        gap_simple_fit["rnd_seed"] = committee_extra_seeds[i_committee - 1]

        # perform the fit
        stdout_file = run_dir / f'stdout.{GAP_name}{committee_name_postfix}{i_committee}.gap_fit'
        run_gap_fit(database_ci, gap_simple_fit, stdout_file=stdout_file, verbose=verbose,
                    remote_info=remote_info, remote_label=remote_label)

        GAP_xml_modify_label(GAPfile, new_label=GAPname)

    print('final GAPs in \'{}\''.format(final_GAPfiles))

    if num_committee > 0:
        return final_GAPfiles, final_GAPnames
    else:
        return final_GAPfiles[0], final_GAPnames[0]


def GAP_xml_modify_label(GAPfile, new_label=None):
    """fix internal GAP name and return updated label

    Parameters
    ----------
    GAPfile: str
        gap xml file
    new_label: str / None, default None
        new label, if None then old label is returned

    Returns
    -------
    label: str
        new label or old if not changed

    """
    et = cElementTree.parse(GAPfile)
    root = et.getroot()
    pot = et.find('Potential')

    if new_label is not None:
        # change outermost XML container name
        root.tag = new_label
        # change first Potential label
        pot.set('label', new_label)
        # rewrite
        et.write(GAPfile)
        return new_label
    else:
        old_label = pot.get('label')
        return old_label
