import numpy as np
from ase.build import molecule
from wfl.configset import OutputSpec, ConfigSet

import pytest

# tested modules
from wfl.generate import radicals
from wfl.generate import smiles

# wfl.generate_configs.smiles depends on rdkit.Chem
pytest.importorskip("rdkit.Chem")


def test_smi_to_atoms():
    """test for wfl.generate_configs.smiles"""

    smi = 'C'
    atoms = smiles.smi_to_atoms(smi)

    assert np.all(atoms.symbols == 'CH4')


def test_smiles_run_op():
    """test for wfl.generate_configs.smiles"""

    input_smiles = ['C', 'C(C)C']
    extra_info = {'config_type': 'testing', 'info_int': 5}

    atoms = smiles.run_op(input_smiles, extra_info=extra_info)

    for smi, at in zip(input_smiles, atoms):

        assert at.info['smiles'] == smi, f'{at.info["smiles"]} doesn\'t match {smi}'

        for key, val in extra_info.items():
            assert at.info[key] == val, f'info entry {key} ({at.info[key]}) doesn\'t match {val}'

    input_smiles = 'C'
    extra_info = None

    atoms = smiles.run_op(input_smiles, extra_info)

    assert len(atoms) == 1
    assert atoms[0].info['smiles'] == input_smiles


def test_abstract_sp3_hydrogens():
    
    smiles_list = ['C', 'C=CCO']
    ConfigSet(input_configs=[smiles.smi_to_atoms(smi) for smi in smiles_list])
    cfs_out = OutputSpec()

    expected_formuli = [['CH3']*4,
                         ['C3OH5']*2]

    expected_config_types = [['rad1', 'rad2', 'rad3', 'rad4'],
                             ['rad7', 'rad8']]

    for mol, ref_formuli, ref_cfg_types in zip(cfs_out.output_configs,
                                               expected_formuli,
                                               expected_config_types):

        rads = radicals.abstract_sp3_hydrogen_atoms(mol)
        formuli = [str(at.symbols) for at in rads]
        assert np.all(formuli == ref_formuli)
        config_types = [at.info['config_type'] for at in rads]
        assert np.all(config_types == ref_cfg_types)

