"""Communly used and/or examples of classes that can be used to abort an MD sampling
run under specified conditions
"""

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
        how many steps in a row any atom pairs have to be too cloe
    """

    def __init__(self, collision_radius, n_failed_steps=3):
        super().__init__(n_failed_steps)
        self.collision_radius = collision_radius


    def atoms_ok(self, at):
        i = neighbor_list('i', at, self.collision_radius)

        if len(i) > 0:
            return False
        else:
            return True
