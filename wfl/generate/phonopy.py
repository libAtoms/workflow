import numpy as np
from ase.atoms import Atoms

from wfl.autoparallelize import _autoparallelize_ll

try:
    import phonopy
except:
    phonopy = None

try:
    import phono3py
except:
    phono3py = None


def run(inputs, outputs, displacements, strain_displs, ph2_supercell, ph3_supercell=None, pair_cutoff=None, num_inputs_per_python_subprocess=10):
    return _autoparallelize_ll(iterable=inputs, outputspec=outputs, op=run_autopara_wrappable, num_inputs_per_python_subprocess=num_inputs_per_python_subprocess, 
                         displacements=displacements, strain_displs=strain_displs, ph2_supercell=ph2_supercell,
                         ph3_supercell=ph3_supercell, pair_cutoff=pair_cutoff)


def run_autopara_wrappable(atoms, displacements, strain_displs, ph2_supercell, ph3_supercell=None, pair_cutoff=None):
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
    ConfigSet corresponding to outputs


    """
    def _sc_to_mat(sc):
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

    ph2_sc_mat = _sc_to_mat(ph2_supercell)
    ph3_sc_mat = _sc_to_mat(ph3_supercell)

    ats_pert = []

    for at0 in atoms:
        at_ph = phonopy.structure.atoms.PhonopyAtoms(numbers=at0.numbers, positions=at0.positions, cell=at0.cell)

        ats_pert.append([])
        for displ_i, displ in enumerate(displacements):
            # need to create Phono[3]py inside loop since generate_displacements() is called once it cannot be
            # called again, apparently

            d2 = []
            d3 = []
            if ph2_sc_mat is not None:
                ph2 = phonopy.Phonopy(at_ph, ph2_sc_mat)
                ph2.generate_displacements(distance=displ)
                d2_undispl = ph2.supercell
                d2 = ph2.supercells_with_displacements

                if displ_i == 0:
                    # store undisplaced phonon (harmonic) config
                    at_pert = Atoms(cell=d2_undispl.cell, positions=d2_undispl.positions, numbers=d2_undispl.numbers, pbc=[True]*3)
                    if ph3_sc_mat is not None and np.any(ph2_sc_mat != ph3_sc_mat):
                        # fc2 and fc3 are different supercells, need different undisplaced configs
                        at_pert.info["config_type"] = f"phonon_harmonic_undispl"
                    else:
                        at_pert.info["config_type"] = f"phonon_undispl"
                    ats_pert[-1].append(at_pert)
                for at in d2:
                    at_pert = Atoms(cell=at.cell, positions=at.positions, numbers=at.numbers, pbc=[True]*3)
                    at_pert.info["config_type"] = f"phonon_harmonic_{displ_i}"
                    ats_pert[-1].append(at_pert)

            if ph3_sc_mat is not None:
                ph3 = phono3py.Phono3py(at_ph, supercell_matrix=ph3_sc_mat)
                ph3.generate_displacements(distance=displ, cutoff_pair_distance=pair_cutoff)
                d3_undispl = ph3.supercell
                d3 = [a for a in ph3.supercells_with_displacements if a is not None]

                if displ_i == 0 and (ph2_sc_mat is None or np.any(ph2_sc_mat != ph3_sc_mat)):
                    # store undisplaced fc3 (cubic) config
                    at_pert = Atoms(cell=d3_undispl.cell, positions=d3_undispl.positions, numbers=d3_undispl.numbers, pbc=[True]*3)
                    if ph2_sc_mat is not None and np.any(ph2_sc_mat != ph3_sc_mat):
                        at_pert.info["config_type"] = f"phonon_cubic_undispl"
                    else:
                        at_pert.info["config_type"] = f"phonon_undispl"
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
