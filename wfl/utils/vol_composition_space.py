import numpy as np


def composition_space_Zs(ats):
    """Elements from composition space

    Parameters
    ----------
    ats : list(Atoms)

    Returns
    -------
    Zs : list(int)
        set of atomic numbers found, sorted
    """
    Zs = set()
    for at in ats:
        Zs |= set(at.numbers)
    return sorted(list(Zs))


def composition_space_coord(at, fields, composition_Zs=None):
    """Calculate coordinates in vol-composition space

    Parameters
    ----------
    at : Atoms
    fields : list(str)
        fields of atoms objects to find:
        - "_V": volume per atom
        - "_x": compositions, n_elements-1
        - any at.info key which is then divided by the number of atoms
    composition_Zs : list(int)
        atomic numbers of elements for composition space

    Returns
    -------
    coords : list(float)
        coordinates, with volume and n_species-1 dimensions

    """
    coords = []
    for f in fields:
        if f == "_V":
            coords.append(at.get_volume() / len(at))
        elif f == "_x":
            # Zs[1:] because you only need n_types-1 composition fractions to fully determine composition
            coords += [np.sum(at.get_atomic_numbers() == Zi) / len(at) for Zi in composition_Zs[1:]]
        elif f in at.info:
            coords.append(at.info[f] / len(at))
        else:
            raise RuntimeError("Got select_coord field {}, not _V or _x or in at.info".format(f))
    return coords
