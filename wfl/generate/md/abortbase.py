from abc import ABC, abstractmethod
import numpy as np

class AbortBase(ABC):
    """Base class used for checking and aborting MD simulation of `wfl.generate.md.sample()`"""
    def __init__(self, n_failed_steps=1):
        self.history = []
        self.n_failed_steps = n_failed_steps

    @abstractmethod
    def check_if_atoms_ok(self, at):
        """Method to check whether this trajectory step is acceptable. 
           The method must append a boolean to `self.history`, 
           based on which the MD gets aborted. 
           All derived classes must implement this method."""
        ...
    
    def should_stop_md(self, at):
        """Returns a boolean based of which `wfl.generate.md.sample()`
        aborts the simulation. Defaults to aborting if `n_failed_steps` in a row `check_if_atoms_ok()` 
        are evaluated to False. Derrived classes may overwrite this."""
        self.check_if_atoms_ok(at)
        return np.all(np.array(self.history[-self.n_failed_steps:]) == False)


class AbortOnCollision(AbortBase):
    def __init__(self, clash_radius=0.5, n_failed_steps=3):
        super().__init__(n_failed_steps)        
        self.clash_radius = clash_radius

    def check_if_atoms_ok(self, at):
        distances = at.get_all_distances()
        distances[distances == 0] = np.nan
        clashes = distances[distances < self.clash_radius] 
        
        if len(clashes) > 0:
            is_this_ok = False
        else:
            is_this_ok = True
        self.history.append(is_this_ok)

