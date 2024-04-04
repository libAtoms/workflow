"""Communly used and/or examples of classes that can be used to abort an MD sampling
run under specified conditions
"""

import numpy as np

from .abort_base import AbortSimBase
from ase.neighborlist import neighbor_list


class AbortOnCollision(AbortSimBase):
    """Abort an MD run if a collision (two atoms closer than some distance) happens
    for a number of steps in a row

    Parameters
    ----------
    collision_radius: float
        distance for atoms to be considered a collision

    n_failed_steps: int, default 1
        how many steps in a row any atom pairs have to be too close
    """

    def __init__(self, collision_radius, n_failed_steps=3):
        super().__init__(n_failed_steps)
        self.collision_radius = collision_radius


    def atoms_ok(self, atoms):
        i = neighbor_list('i', atoms, self.collision_radius)

        if len(i) > 0:
            return False
        else:
            return True


class AbortOnLowEnergy(AbortSimBase):
    """Abort an MD run if the energy drops by too much

    Parameters
    ----------
    delta_E_per_atom: float
        drop in energy per atom to trigger abort
    """

    def __init__(self, delta_E_per_atom):
        super().__init__(1)
        self.delta_E_per_atom = -np.abs(delta_E_per_atom)
        self.initial_E_per_atom = None


    def atoms_ok(self, atoms):
        E_per_atom = atoms.get_potential_energy() / len(atoms)
        if self.initial_E_per_atom is None:
            self.initial_E_per_atom = E_per_atom
            return True
        else:
            return (E_per_atom - self.initial_E_per_atom) >= self.delta_E_per_atom
