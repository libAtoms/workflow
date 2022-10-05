import numpy as np
from ase.build import molecule

import pytest

# tested modules
from wfl.generate import smiles

# wfl.generate_configs.smiles depends on rdkit.Chem
pytest.importorskip("rdkit.Chem")


def test_smi_to_atoms():
    """test for wfl.generate_configs.smiles"""

    smi = 'C'
    atoms = smiles.smi_to_atoms(smi)

    assert np.all(atoms.symbols == 'CH4')


def test_smiles_run_autopara_wrappable():
    """test for wfl.generate_configs.smiles"""

    input_smiles = ['C', 'C(C)C']
    extra_info = {'config_type': 'testing', 'info_int': 5}

    atoms = smiles._run_autopara_wrappable(input_smiles, extra_info=extra_info)

    for smi, at in zip(input_smiles, atoms):

        assert at.info['smiles'] == smi, f'{at.info["smiles"]} doesn\'t match {smi}'

        for key, val in extra_info.items():
            assert at.info[key] == val, f'info entry {key} ({at.info[key]}) doesn\'t match {val}'

    input_smiles = 'C'
    extra_info = None

    atoms = smiles._run_autopara_wrappable(input_smiles, extra_info)

    assert len(atoms) == 1
    assert atoms[0].info['smiles'] == input_smiles
