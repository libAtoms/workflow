#!/usr/bin/env python3

import json
import os
import pathlib
from pprint import pprint

import click
import numpy as np
from ase.units import GPa

from wfl.calculators.dft import evaluate_dft
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.buildcell import buildcell as run_buildcell


@click.command('')
@click.argument("buildcell-inputs", nargs=-1)
@click.option('--verbose', '-v', is_flag=True)
@click.option('--configuration', '-c', type=click.STRING, required=True,
              help="GAP-RSS config file, only reads the `DFT_evaluate` section")
@click.option('--buildcell-cmd', '-b', type=click.STRING, default='buildcell', envvar='GRIF_BUILDCELL_CMD')
@click.option('--n-per-config', '-n', type=click.INT, default=5)
@click.option('--param-ranges', '-r', type=click.STRING, required=True,
              help="JSON string with ranges of for parameters to test")
def cli(verbose, configuration, buildcell_inputs, buildcell_cmd, n_per_config, param_ranges):
    """DFT parameter convergence testing for CASTEP and VASP.
    """
    if len(buildcell_inputs) == 0:
        raise RuntimeError('Need buildcell_inputs')

    output_prefix = "CONVERGE_"
    run_dir = 'run_dft_convergence_test'
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

    with open(configuration) as fin:
        params = json.load(fin)['DFT_evaluate']

    # do buildcell
    structs = []
    for filename in buildcell_inputs:
        with open(filename) as fin:
            buildcell_input = fin.read()
        c_out = OutputSpec(f'structs.{filename}.xyz', file_root=run_dir),
        structs.append(run_buildcell(c_out, range(n_per_config), buildcell_cmd=buildcell_cmd,
                                     buildcell_input=buildcell_input, verbose=verbose))
    # merge
    structs = ConfigSet(structs)

    # get ranges of various keyword arguments
    ranges = {}
    dft_evaluated = {}
    for key, arange_args in json.loads(param_ranges).items():
        ranges[key] = np.arange(*arange_args)
        dft_evaluated[key] = {}

    if verbose:
        print("Running test of the following parameters:")
        pprint(ranges)

    for key, param_range in ranges.items():
        for param_val in param_range:
            run_kwargs = params.get('kwargs', {}).copy()
            # reset to minimal values
            for k, r in ranges.items():
                run_kwargs[k] = r[0]
            # set desired value
            run_kwargs[key] = param_val

            evaluated_structs = OutputSpec(f'DFT_evaluated.{key}_{param_val}.xyz', file_root=run_dir)

            dft_evaluated[key][param_val] = evaluate_dft(
                inputs=structs, outputs=evaluated_structs,
                calculator_name=params['calculator'],
                workdir_root=os.path.join(run_dir, 'vasp_run'),
                calculator_kwargs=run_kwargs,
                output_prefix=output_prefix,
                keep_files='default' if verbose else False)

    print('E in eV/atom, F in eV/A, stress in GPa')
    print('"dX" is maximum over configs, atoms, and components, "mean" is mean absolute values over same set')
    num_structures = len(buildcell_inputs) * n_per_config
    for key, param_range in ranges.items():

        # filter the structures for which EVERY calculation succeeded
        keep_atoms = np.ones(shape=num_structures, dtype=bool)
        for atoms_list in dft_evaluated[key].values():
            succeeded = [f"{output_prefix}energy" in at.info.keys() for at in atoms_list]
            keep_atoms = np.logical_and(keep_atoms, succeeded)

        print(f'\nCONVERGENCE WITH RESPECT TO {key} ({sum(keep_atoms)} / {num_structures} succeeded)')
        es = []
        fs = []
        ss = []
        for param_val, ats in dft_evaluated[key].items():
            e_f_s = [(at.info[f"{output_prefix}energy"] / len(at),
                      at.arrays[f"{output_prefix}forces"],
                      at.info[f"{output_prefix}stress"]) for i, at in enumerate(ats) if keep_atoms[i]]

            es.append(np.asarray([e[0] for e in e_f_s]))
            fs.append(np.asarray([fv for f in e_f_s for fv in f[1].flatten()]))
            ss.append(np.asarray([sv for s in e_f_s for sv in s[2].flatten()]))
        keys = list(dft_evaluated[key].keys())
        print('{:21s} E {:>7s} / {:>7s} F {:>7s} / {:>7s} S {:>7s} / {:>7s}'.format('prev - cur_param', 'dE', 'mean(E)',
                                                                                    'dF', 'mean(F)', 'dS', 'mean(S)'))
        for param_val, param_prev, e_val, e_prev, f_val, f_prev, s_val, s_prev in zip(
                keys[1:], keys[0:-1], es[1:], es[0:-1], fs[1:], fs[0:-1], ss[1:], ss[0:-1]):
            mean_e = np.mean(np.abs(e_val))
            de = np.max(np.abs(e_val - e_prev))
            mean_f = np.mean(np.abs(f_val))
            df = np.max(np.abs(f_val - f_prev))
            mean_s = np.mean(np.abs(s_val)) / GPa
            ds = np.max(np.abs(s_val - s_prev)) / GPa
            print('{:10.4f}-{:10.4f} E {:7.3f} / {:7.3f} F {:7.3f} / {:7.3f} S {:7.3f} / {:7.3f}'.format(param_prev,
                                                                                                         param_val, de,
                                                                                                         mean_e, df,
                                                                                                         mean_f, ds,
                                                                                                         mean_s))


if __name__ == '__main__':
    cli()
