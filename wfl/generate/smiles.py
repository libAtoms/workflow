import io

from ase.io import read
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ModuleNotFoundError:
    Chem = None

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring


def smi_to_atoms(smi, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, randomSeed=-1):
    """Converts smiles to 3D Atoms object"""
    if Chem is None:
        raise RuntimeError("rdkit must be installed for SMILES support")

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    _ = AllChem.EmbedMolecule(mol, useBasicKnowledge=useBasicKnowledge,
                              useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
                              randomSeed=randomSeed)

    insert = 'Properties=species:S:1:pos:R:3'
    xyz = Chem.rdmolfiles.MolToXYZBlock(mol)
    xyz = xyz.split(sep='\n\n')
    xyz = f'{xyz[0]}\n{insert}\n{xyz[1]}'
    xyz_file = io.StringIO(xyz)

    atoms = read(xyz_file, format='xyz')
    return atoms



def _run_autopara_wrappable(smiles, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, extra_info=None,
                            randomSeed=-1):
    """Creates atomic configurations by repeatedly running smi_to_xyz, I/O with OutputSpec.

    Parameters
    ----------
    smiles: str/list(str)
       smiles string to generate structure from
    useBasicKnowledge: bool, default True
        impose basic knowledge such as flat aromatic rings
    useExpTorsionAnglePrefs: bool, default True
        impose experimental torsion angle preferences
    extra_info: dict, default {}
        extra fields to place into created atoms info dict
    randomSeed: int, default -1
        RDKit EmbedMolecule random seed for reproducibility

    Returns
    -------
    list(Atoms) generated from SMILES

    """

    if extra_info is None:
        extra_info = {}
    if isinstance(smiles, str):
        smiles = [smiles]

    atoms_list = []
    for smi in smiles:
        at = smi_to_atoms(smi=smi, useBasicKnowledge=useBasicKnowledge,
                          useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
                          randomSeed=randomSeed)
        at.info['smiles'] = smi
        for key, value in extra_info.items():
            at.info[key] = value
        atoms_list.append(at)

    return atoms_list


def smiles(*args, **kwargs):
    if Chem is None:
        raise RuntimeError("rdkit must be installed for SMILES support")
    return autoparallelize(_run_autopara_wrappable, *args, **kwargs)
autoparallelize_docstring(smiles, _run_autopara_wrappable, "SMILES string")
