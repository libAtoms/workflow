from abc import ABC, abstractmethod
import numpy as np


class AbortSimBase(ABC):
    """Base class used for checking and aborting MD simulation of `wfl.generate.md.sample()`.
    See `stop` method docstring for its default behavior.
    """
    def __init__(self, n_failed_steps=1):
        self.ok_history = []
        self.n_failed_steps = n_failed_steps


    @abstractmethod
    def atoms_ok(self, at):
        """Method returning a boolean indicating whether this trajectory step is acceptable.
           All derived classes must implement this method.

        Parameters
        ----------

        at: Atoms
            atomic configuration

        Returns
        -------
        is_ok: bool containing status
        """
        ...


    def stop(self, at):
        """Returns a boolean indicating whether `wfl.generate.md.sample()` should stop
        the simulation. Defaults to aborting if `n_failed_steps` in a row `atoms_ok()`
        are evaluated to False. Derrived classes may overwrite this."""
        self.ok_history.append(self.atoms_ok(at))
        return np.all(np.logical_not(np.array(self.ok_history[-self.n_failed_steps:])))
