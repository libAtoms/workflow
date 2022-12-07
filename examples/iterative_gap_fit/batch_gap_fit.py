'''
iterative GAP fitting of energies for Cu bulk & slabs with EMT calculator
'''
import os

os.environ['WFL_NUM_PYTHON_SUBPROCESSES'] = "4"
os.environ['WFL_GAP_FIT_OMP_NUM_THREADS'] = "4"


import wfl.autoparallelize
wfl.autoparallelize.mpipool_support.init(verbose=4)

import json, os, yaml
import numpy as np

from ase.io import read, write
from ase.calculators.emt import EMT

from pathlib import Path

from quippy.potential import Potential

from wfl.calculators.generic import run as generic_calc
from wfl.descriptors.quippy import from_any_to_Descriptor
from wfl.descriptors.quippy import calc as desc_calc
from wfl.configset import ConfigSet, OutputSpec
from wfl.fit.gap.multistage import prep_params
from wfl.fit.gap.multistage import fit as gap_fit
from wfl.fit.error import calc as ref_error_calc
from wfl.generate.md import sample as sample_md
from wfl.generate.optimize import run as optimize
from wfl.select.by_descriptor import greedy_fps_conf_global


def get_ref_error(in_file, out_file, gap_file, **kwargs):
    """
    Calculate the error between your reference calculator and the previously
    trained GAP.

    Parameters
    ----------
    in_file:    str,
        Path to file containing the configs to check
    out_file:   str
        Path to file in which the calculated references will be written.
        If None, these will be stored in the Memory and deleted after the
        function is completed
    gap_file:   str
        Path to GAP parameter file, which is required for creating a
        quippy.Potential class.
    kwargs:
    ref_property_prefix:    str, default='DFT_'
        name of the calculated reference properties. This is what the GAP values
        will be compared to.
    properties:             list, default=['energy_per_atom', 'forces']
        names of the properties to check in the
        wfl.fit.ref_error err_from_calculated_ats function.

    Returns
    -------
    errors: dict
        dictionary containing the errors for the respective properties
    """
    in_config = ConfigSet(in_file)
    out_config = OutputSpec(files=out_file)
    calculator = (Potential, None, {'param_filename': gap_file})
    ref_property_prefix = kwargs.get('ref_property_prefix', 'DFT_')
    category_keys = kwargs.get('category_keys', None)

    calc_in_config = generic_calc(in_config, OutputSpec(), calculator, output_prefix='calc_')
    error, _, _ = ref_error_calc(calc_in_config, 'calc_', ref_property_prefix,
                                 category_keys=category_keys, config_properties=["energy/atom"], atom_properties=["forces"])

    error = {'energy': error['energy/atom']['_ALL_']['RMSE'],
             'forces': error['forces']['_ALL_']['RMSE']}
    return error


def run_md(atoms, out_file, gap_file, **kwargs):
    """
    Generates new configs via the wfl.generate_configs.md sample function.

    Parameters
    ----------
    md.sample runs an MD and samples structures based on dt and steps,
    Next step is implementing furthest point sampling
    (Check md.py file in wfl -> generate_configs for details)
    parameters to set MD can be defined as arguments
    (NPT/NVT, temp, pressure, etc)
    """
    in_config = ConfigSet(atoms)
    out_config = OutputSpec(files=out_file)
    calculator = (Potential, None, {'param_filename': gap_file})
    sample_md(in_config, out_config, calculator=calculator, **kwargs)
    return None


def run_optimize(atoms, out_file, gap_file, **kwargs):
    """
    Generates new configs via the wfl.generate_configs.optimize run function.

    Parameters
    ----------
    atoms:  list
        list of configs to be relaxed.
        #IMPORTANT: Constraints, such as fixed atoms need to be set previously
    out_file: str
        file in which the relaxation trajectories will be stored
    gap_file: str
        Path to GAP parameter file which we will use to create the calculator

    **kwargs:
    In general: kwargs for the run function, the two "main" ones:
        "fmax": float, the force convergence criteria for the relaxation
        "steps": int, maximum permissible number of steps during the relaxation
    """
    in_config = ConfigSet(atoms)
    out_config = OutputSpec(files=out_file)
    calculator = (Potential, None, {'param_filename': gap_file})
    optimize(in_config, out_config, calculator=calculator, **kwargs)
    return None


def run_emt(in_file, out_file, **kwargs):
    """
    Runs the ASE EMT calculator via the wfl.calcualtors.generic run function.

    Parameters
    ----------
    in_file:    str,
        Path to file containing the configs to check
    out_file:   str
        Path to file in which the EMT-calculated configs will be written.
        If None, these will be stored in the Memory and deleted after the
        function is completed
    kwargs
    properties:     list, default=['energy', 'forces']
        list of properties to calculate
    output_prefix:  str, default='DFT_'
        assigned prefix for all EMT calculated properties
    Returns
    -------
    """
    in_config = ConfigSet( in_file)
    out_config = OutputSpec(files = out_file)
    calculator = (EMT, None, {"fixed_cutoff": True})
    properties = kwargs.pop("properties", ["energy", "forces"])
    output_prefix = kwargs.pop("output_prefix", "DFT_")
    generic_calc(in_config, out_config, calculator,
        properties=properties, output_prefix=output_prefix, **kwargs
    )
    return None


def get_descriptors(in_file, out_file, params_file, key='desc', **kwargs):
    """
    Uses a GAP descriptor parameter file to calculate the global descriptor
    vector for each config within a given set of configs.

    Parameters
    ----------
    in_file:        str,
        Path to file containing the configs to check
    out_file:       str
        Path to file in which the descriptors are added to the info
        If None, these will be stored in the Memory and deleted after the
        function is completed
    params_file:    str
        Path to GAP descriptor parameter file, from here we will take the SOAP
        vectors that are required to calculate globally averaged descriptor
        vectors
    key:            str, default='desc'
        config's descriptor label in the config.info dictionary
    kwargs
        Any kwarg that is in the wfl.calc_descriptor calc function
    Returns
    -------
    None, the descriptors are going to be found in the out_file
    """
    in_config = ConfigSet(in_file)
    out_config = OutputSpec(files = out_file)

    params = yaml.safe_load(open(params_file, 'r'))
    params = [i for i in params if 'soap' in i.keys()]

    per_atom = True
    for param in params:
        if 'average' not in param.keys():
            param['average'] = True
            per_atom = False

    # Calculate the actual descriptors (gets added as key in at.info)
    desc_calc(in_config, out_config, params, key, per_atom, **kwargs)
    return None


def run_fps(in_file, out_file, n_samples, **kwargs):
    """
    Runs global farthest point sampling implemented via the
    wfl.select_configs.by_descriptor greedy_fps_conf_global function

    Parameters
    ----------
    in_file:        str,
        Path to file containing the configs to check
    out_file:       str
        Path to file in which the selected configs are written
        If None, these will be stored in the Memory and deleted after the
        function is completed
    n_samples:    int
        Number of configs to select
    kwargs
    training_desc_file: str, default=False
        path to file containing descriptors of the training configs
        If you don't include these the fps will simply find the fps within the
        in_file.
    prev_descs:         list, default=[]
        Alternative to training_desc_file, list of descriptor vectors that
        the in_configs will be compared to
    keep_descriptors:   bool, default=False
        If False, descriptors are immediately deleted after fps is complete
    desc_key:           str, default='desc'
        Label of descriptors to compare in configs.info
    Returns
    -------
    None, the selected configs are written in the out_file
    """
    in_config = ConfigSet(in_file)
    out_config = OutputSpec(files = out_file)

    tdf = kwargs.pop('training_desc_file', False)
    if tdf:
        prev_descs = [i.info.get('desc') for i in read(tdf, ':')]
    else:
        prev_descs = kwargs.pop('prev_descs', [])

    keep_descriptors = kwargs.pop('keep_descriptor_info', False)
    desc_key = kwargs.pop('desc_key', 'desc')

    greedy_fps_conf_global(in_config, out_config, n_samples,
        at_descs_info_key=desc_key, keep_descriptor_info = keep_descriptors,
        prev_selected_descs = prev_descs,**kwargs
    )
    return None


def get_gap(in_file, gap_name, Zs, length_scales, params,
        ref_property_prefix='DFT_', run_dir='GAP'
    ):
    """
    Run the wfl.fit.gap_multistage fit function.

    Parameters
    ----------
    in_file:                str
        Path to file containing the input configs for the GAP fit
    gap_name:               str
        File name of written GAP
    Zs:                     list
        List of atomic numbers in the GAP fit.
    length_scales:          dict
        Length scale dictionary for each atomic species in the fit.
        Dictionary keys are the atomic numbers, values are dictionaries that
        must contain the keys "bond_len" and "min_bond_len"
    params:                 dict
        GAP fit parameters, see the parameter json files for more information
    ref_property_prefix:    str, default="DFT_"
        label prefixes for the in_config properties.
    run_dir:                str, default='GAP'
        Name of the directory in which the GAP files will be written
    Returns
    -------
    None, the selected configs are written in the out_file
    """
    in_config = ConfigSet(in_file)
    gap_params = prep_params(Zs, length_scales, params)
    gap_fit(in_config, gap_name, gap_params,
        run_dir=run_dir, ref_property_prefix=ref_property_prefix
    )
    return None


def get_file_names(GAP_dir, MD_dir, fit_idx, calc='md'):
    """
    Function to keep track of the different file locations
    """
    files = {}
    f1 = lambda x, y: f"{MD_dir}/{x}{y}_{fit_idx}.xyz"
    f2 = lambda x: f"{GAP_dir}/GAP_{fit_idx}.xml{x}"

    if calc == 'md':
        files["calc_out"] = f1("md", '')
    elif calc == 'optimize':
        files['calc_out'] = f1("optimize", '')
    files["desc"]   = f1("desc", '')
    files["fps"]    = f1("fps", '')
    files["dft"]    = f1("dft", '')
    files["eval"]   = f1("eval", '')
    files["training_desc"] = f1("training_desc", "")

    files["gap"]        = f2("")
    files["gap_params"] = f2(".descriptor_dicts.yaml")
    return files


def main(max_count=5, verbose=False):
    workdir = os.path.join(os.path.dirname(__file__))

    ### GAP parameters
    gap_params = os.path.join(workdir, 'multistage_gap_params.json')
    with open(gap_params, 'r') as f:
        gap_params = json.loads(f.read())
    Zs = [29]
    length_scales = {
            29: {
                "bond_len": [2.6, "NB VASP auto_length_scale"],
                "min_bond_len": [2.2, "NB VASP auto_length_scale"],
                "other links": {},
                "vol_per_atom": [12, "NB VASP auto_length_scale"]
                }
    }

    training = os.path.join(workdir, 'EMT_atoms.xyz')

    ### Initial GAP training
    fit_idx = 0
    gap_name = f'GAP_{fit_idx}'
    GAP_dir = Path(os.path.join(workdir, 'GAP'))
    GAP_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"Fitting original GAP located in {GAP_dir}/{gap_name}.xml",
            flush=True)
    get_gap(training, gap_name, Zs, length_scales, gap_params, run_dir=GAP_dir)

    ### MD info
    calc = 'md'
    MD_dir = Path(os.path.join(workdir, 'MD'))
    MD_dir.mkdir(exist_ok=True)
    md_in_file = os.path.join(workdir, 'init_md.traj')
    md_configs = read(md_in_file, ':')
    md_params = {'steps': 300, 'dt': 1, 'temperature': 450.}

    '''
    ### optimize Info
    calc = 'optimize'
    optimize_params = {
            "steps": 50,
            "fmax": 0.001,
            "pressure": None,
            "keep_symmetry": False,
            "verbose": True,
        }
    #'''

    n_select = 15 # The number of selected configs from each generation
    print(f"Maximum number of generation iterations: {max_count}", flush=True)

    if verbose:
        print(f"Starting the iterative fitting process!", flush=True)

    while fit_idx  < max_count:
        if verbose:
            print(f"Fit idx: {fit_idx}", flush=True)
        files = get_file_names(GAP_dir, MD_dir, fit_idx, calc = calc)
        train_error = get_ref_error(training, None, files["gap"])

        if calc == 'md':
            # Run an MD to create new structures
            run_md(md_configs, files["calc_out"], files["gap"], **md_params)
        elif calc == 'optimize':
            # Run an ase relaxation, to create new structures.
            run_optimize(md_configs, files["calc_out"], files["gap"], **optimize_params)

        # Calculate the descriptors for the md output & sample them via fps
        get_descriptors(files["calc_out"], files["desc"], files["gap_params"])
        get_descriptors(training, files["training_desc"], files["gap_params"])
        run_fps(files["desc"], files["fps"], n_select,
            training_desc_file=files["training_desc"]
        )
        run_emt(files["fps"], files["dft"])

        val_error   = get_ref_error(files["dft"], files["eval"], files["gap"])

        v_f, v_e = 1000 * val_error['forces'], 1000 * val_error['energy']
        t_f, t_e = 1000 * train_error['forces'], 1000 * train_error['energy']

        if verbose:
            log_dict = {
                "fit_idx": fit_idx,
                "Validation: RMSE_f": v_f, "Training: RMSE_f": t_f,
                "Validation: RMSE_e": v_e, "Training: RMSE_e": t_e
            }

            f_dev = abs(100 * (v_f - t_f)/t_f)
            e_dev = abs(100 * (v_e - t_e)/t_e)

            print(
                f'VALIDATION: RMSE Forces: {v_f:.2f}, RMSE Energy: {v_e:.2f}\n'
                f'TRAINING:   RMSE Forces: {t_f:.2f}, RMSE Energy: {t_e:.2f}\n'
                f'DEVIATIONS: Forces:{f_dev:.2f}%, Energy: {e_dev:.2f}%',
                flush=True
            )
            with open(os.path.join(workdir, 'errors.json'), 'a') as f:
                json.dump(log_dict, f)
                f.write('\n')

        fit_idx += 1
        training_atoms = read(training, ':') + read(files["dft"], ':')
        training = f'{GAP_dir}/training_{fit_idx}.xyz'
        gap_name = f'GAP_{fit_idx}'
        write(training, training_atoms)
        get_gap(training, gap_name, Zs, length_scales, gap_params,
            run_dir=GAP_dir
        )
    return None


if __name__ == '__main__':
    main(verbose=True)
