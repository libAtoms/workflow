import os
import numpy as np
import subprocess
from pathlib import Path
import shlex
import warnings

from ase.stress import voigt_6_to_full_3x3_stress

from wfl.utils.julia import julia_exec_path


def ace_fit_jl_path():
    if "WFL_ACE_FIT_COMMAND" in os.environ:
        return os.environ["WFL_ACE_FIT_COMMAND"]

    julia_exec = julia_exec_path()

    ace_path_input = "import(ACE1pack)\nprint(pathof(ACE1pack))\n"
    try:
        ace_path = subprocess.check_output(shlex.split(julia_exec), text=True, shell=True, stderr=subprocess.STDOUT,
                                           input=ace_path_input)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to find path of ACE1pack") from exc

    # some julia errors don't lead to a CalledProcessError
    if len(ace_path.strip().split("\n")) > 1 or "Stacktrace:" in ace_path or "ERROR:" in ace_path:
        raise RuntimeError("Failed to get ace_path\n" + ace_path)

    ace_path = Path(ace_path)
    ace_path = ace_path.resolve().parent.parent

    ace_fit_command = julia_exec + " " + str(ace_path / "scripts" / "ace_fit.jl")

    warnings.warn(f"Automatically found ace fit command {ace_fit_command}")
    return ace_fit_command


def fix_stress_virial(configs, ref_property_keys, stress_key):
    """convert from stress to 3x3 virial, workaround until fitting code accepts stress

    Parameters
    ----------
    configs: list(Atoms)
        fitting configurations
    ref_property_keys: dict
        dict with each ASE property key as keys and corresponding info/array keys as values
    stress_key: str
        key for stress property
    """
    if stress_key is not None:
        for at in configs:
            if stress_key in at.info:
                stress = at.info[stress_key]
                # delete so as not to confuse gap_fit, until gap_fit is updated to fit to stress directly
                del at.info[stress_key]

                if ref_property_keys['virial'] in at.info:
                    # virial overrides stress if present
                    continue

                virial = - stress * at.get_volume()
                if len(virial) == 6:
                    virial = voigt_6_to_full_3x3_stress(virial)
                at.info[ref_property_keys['virial']] = virial

    # reshape virial into 9-vector so Fortran can read it, since only if it
    # has a special name will ase.io.write know to do that
    if ref_property_keys['virial'] is not None:
        for at in configs:
            if ref_property_keys['virial'] in at.info:
                at.info[ref_property_keys['virial']] = np.reshape(at.info[ref_property_keys['virial']], 9)


def copy_properties(configs, ref_property_keys, stress_to_virial=True, force=True):
    """Copies properties from wherever they are to 'CALC_*' where fitting code will be told to look for them

    Parameters
    ----------
    configs: list(Atoms)
        fitting configurations
    ref_property_keys: str / dict / None, default None
        where to find properties: str used as a prefix to ASE properties 'energy', 'forces', 'virial',
        'hessian'; dict with each of those as key and actual info/arrays key as value, and None for
        attached calculator (e.g. SinglePointCalculator)
    stress_to_virial: bool, default True
        copy from stress to virial
    forces: bool, default True
        overwrite existing info/arrays items


    Returns
    -------
    ref_property_keys: dict
        each ASE property as key and corresponding info/array key as value
    """
    # make sure that property keys are set up properly, copying from (SinglePoint?) Calculator if needed
    if isinstance(ref_property_keys, str):
        # to convert from stress to virial
        stress_key = ref_property_keys + 'stress'
        # prefix
        ref_property_keys = {k: ref_property_keys + k for k in ['energy', 'forces', 'virial', 'hessian']}
    elif ref_property_keys is None:
        # get from calculator (into SinglePointCalculator SPC), so below all quantities can be addressed via info/arrays
        for at in configs:
            for result_key in ['energy', 'virial', 'stress']:
                if result_key in at.calc.results:
                    if not force and f'CALC_{result_key}' in at.info:
                        raise RuntimeError(f'at.info key CALC_{result_key} already exists)')
                    at.info[f'CALC_{result_key}'] = at.calc.results[result_key]
            if 'forces' in at.calc.results:
                if 'CALC_forces' in at.arrays:
                    if force:
                        del at.arrays['CALC_forces']
                    else:
                        raise RuntimeError('at.arrays key CALC_forces already exists')
                at.new_array('CALC_forces', at.calc.results['forces'])
            if 'CALC_hessian' in at.arrays:
                # no Hessian (in the format we need) from calculator, remove any old ones
                del at.arrays['CALC_hessian']

        ref_property_keys = {k: 'CALC_' + k for k in ['energy', 'forces', 'virial']}
        # to convert from stress to virial
        stress_key = 'CALC_stress'
    elif isinstance(ref_property_keys, dict):
        # check for correct keys in dict
        for k in ['energy', 'forces', 'virial', 'hessian']:
            if k not in ref_property_keys:
                raise RuntimeError('ref_property_keys dict missing one of energy, forces, hessian')
        stress_key = ref_property_keys.get('stress', None)
        if 'stress' in ref_property_keys:
            del ref_property_keys['stress']
    else:
        raise RuntimeError('Got ref_property_keys of unknown type \'{}\''.format(type(ref_property_keys)))

    if stress_to_virial:
        fix_stress_virial(configs, ref_property_keys, stress_key)

    return ref_property_keys
