import numpy as np
from ase.atoms import Atoms

from wfl.pipeline import iterable_loop

try:
    import phonopy
except:
    phonopy = None

try:
    import phono3py
except:
    phono3py = None


def run(inputs, outputs, displacements, strain_displs, ph2_supercell, ph3_supercell=None, pair_cutoff=None, chunksize=10):
    return iterable_loop(iterable=inputs, configset_out=outputs, op=run_op, chunksize=chunksize, 
                         displacements=displacements, strain_displs=strain_displs, ph2_supercell=ph2_supercell,
                         ph3_supercell=ph3_supercell, pair_cutoff=pair_cutoff)


def run_op(atoms, displacements, strain_displs, ph2_supercell, ph3_supercell=None, pair_cutoff=None):
    """create displaced configs with phonopy or phono3py for each structure in inputs

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    displacements: list(float)
        list of displacement magnitudes to use
    strain_displs: list(float)
        list of deformation gradient magnitudes to use
    ph2_supercell: int or int 3-vector or 3x3 matrix
        supercell to use for harmonic force constant displacements
    ph2_supercell: int or int 3-vector or 3x3 matrix, default None
        if not None, supercell to use for cubic force constant displacements
    pair_cutoff: float, default None
        if not None, max distance between pairs of atoms displaced for cubic force constants

    Returns
    -------
    ConfigSet_in corresponding to outputs


    """
    def sc_to_mat(sc):
        if sc is None:
            return None

        if isinstance(sc, int):
            return sc * np.eye(3)

        sc = np.asarray(sc)
        if len(sc.shape) == 1:
            assert sc.shape == (3,)
            sc_mat = np.diag(sc)
        else:
            assert sc.shape == (3,3)
            sc_mat = sc

        return sc_mat

    ph2_sc_mat = sc_to_mat(ph2_supercell)
    ph3_sc_mat = sc_to_mat(ph3_supercell)

    ats_pert = []

    for at0 in atoms:
        at_ph = phonopy.structure.atoms.PhonopyAtoms(numbers=at0.numbers, positions=at0.positions, cell=at0.cell)

        ats_pert.append([])
        for displ_i, displ in enumerate(displacements):
            # need to create Phono[3]py inside loop since generate_displacements() is called once it cannot be
            # called again, apparently
            if ph3_supercell is not None:
                ph3 = phono3py.Phono3py(at_ph, supercell_matrix=ph3_sc_mat, phonon_supercell_matrix=ph2_sc_mat)
            else:
                ph2 = phonopy.Phonopy(at_ph, ph2_sc_mat)

            if ph3_supercell is not None:
                ph3.generate_displacements(distance=displ, cutoff_pair_distance=pair_cutoff)
                d2 = ph3.phonon_supercells_with_displacements
                d3 = [a for a in ph3.supercells_with_displacements if a is not None]
            else:
                ph2.generate_displacements(distance=displ)
                d2 = ph2.supercells_with_displacements
                d3 = []

            for at in d2:
                at_pert = Atoms(cell=at.cell, positions=at.positions, numbers=at.numbers, pbc=[True]*3)
                at_pert.info["config_type"] = f"phonon_harmonic_{displ_i}"
                ats_pert[-1].append(at_pert)

            for at in d3:
                at_pert = Atoms(cell=at.cell, positions=at.positions, numbers=at.numbers, pbc=[True]*3)
                at_pert.info["config_type"] = f"phonon_cubic_{displ_i}"
                ats_pert[-1].append(at_pert)

        for displ_i, displ in enumerate(strain_displs):
            for i0 in range(3):
                for i1 in range(i0+1):
                    F = np.eye(3)
                    F[i0,i1] += displ
                    at_pert = at0.copy()
                    at_pert.set_cell(at_pert.cell @ F, scale_atoms=True)
                    at_pert.info["config_type"] = f"phonon_strain_{displ_i}"
                    ats_pert[-1].append(at_pert)

    return ats_pert
