import io

from ase.io import read
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ModuleNotFoundError:
    pass

from wfl.autoparallelize import _autoparallelize_ll


def smi_to_atoms(smi, useBasicKnowledge=True, useExpTorsionAnglePrefs=True):
    """Converts smiles to 3D Atoms object"""

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    _ = AllChem.EmbedMolecule(mol, useBasicKnowledge=useBasicKnowledge,
                              useExpTorsionAnglePrefs=useExpTorsionAnglePrefs)

    insert = 'Properties=species:S:1:pos:R:3'
    xyz = Chem.rdmolfiles.MolToXYZBlock(mol)
    xyz = xyz.split(sep='\n\n')
    xyz = f'{xyz[0]}\n{insert}\n{xyz[1]}'
    xyz_file = io.StringIO(xyz)

    atoms = read(xyz_file, format='xyz')
    return atoms


def run(outputs, smiles, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, extra_info=None):
    """Creates atomic configurations by repeatedly running smi_to_xyz, I/O with OutputSpec.

    Parameters
    ----------
    outputs: OutputSpec
        where to write outputs
    smiles: str/list(str)
       smiles string to generate structure from
    useBasicKnowledge: bool, default True
        impose basic knowledge such as flat aromatic rings
    useExpTorsionAnglePrefs: bool, default True
        impose experimental torsion angle preferences
    extra_info: dict, default {}
        extra fields to place into created atoms info dict

    Returns
    -------
    ConfigSet corresponding to output

    """

    return _autoparallelize_ll(iterable=smiles, outputspec=outputs, op=run_autopara_wrappable,
                         useBasicKnowledge=useBasicKnowledge,
                         useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
                         extra_info=extra_info)


def run_autopara_wrappable(smiles, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, extra_info=None):
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
                          useExpTorsionAnglePrefs=useExpTorsionAnglePrefs)
        at.info['smiles'] = smi
        for key, value in extra_info.items():
            at.info[key] = value
        atoms_list.append(at)

    return atoms_list
