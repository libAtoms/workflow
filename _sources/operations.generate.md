# Generating Atomic Structures

This page (and submodules of `wfl.generate`) gives a brief overview self-contained operations in Workflow that loosely cover creating new atomic structures or modifying existing ones. All but "atoms and dimers" functions below make use of Workflow's autoparallelization functionality. 

## Atoms and Dimers

`wfl.generate.atoms_and_dimers.prepare()` makes a set of dimer (two-atom) configurations for specified elements and specified distance range. See documentation and example [Generating Dimer Structures](examples.dimers.ipynb).


## SMILES to Atoms

`wfl.generate.smiles.run()` converts [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) (e.g. "CCCC" for n-butane) to ASE's `Atoms`. See example [SMILES to `Atoms`](examples.smiles.md).


## BuildCell

`wfl.generate.buildcell.run()` wrapps [AIRSS](https://airss-docs.github.io/technical-reference/buildcell-manual)'s `buildcell` that constructs sensible random structures. See documentation and example [Random Structures via buildcell](examples.buildcell.ipynb).


## Super Cells

Functions in `wfl.generate.supercells` creates supercells from given primitive cells. These include 

- `largest_bulk()` - makes largest bulk-like supercell with no more than specified number of atoms. 
- `vacancy()` - makes a vacancy in a largest bulk-like supercell from above. 
- `antisite()` - makes antisites in a largest bulk-like supercell from above.  
- `interstitial()` - makes interstitials in a largest bulk-like supercell from above.  
- `surface()` - makes a surface supercell. 


## Molecular Dynamics

Molecular dynamics submodule aimed at sampling atomic configurations. Allows for NVE (Velocity Verlet), NPT (Berendsen) and NVT (Berendsen) integrators. Has hooks for custom functions that sample configs from the trajectory on-the-fly and/or at the end of the individual simulation and also for stopping the simulation early if some condition is met (e.g. MD is unstable). 


## Geometry Optimization

`wfl.generate.optimize.run() optimizes geometry with the given calculator and PreconLBFGS, including symmetry constraints. 


## Minima Hopping

`wfl.generate.minimahopping.run()` wraps ASE's [Minima hopping](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#minima-hopping) code. This algorithm utilizes a series of alternating steps of NVE molecular dynamics and local optimizations. 


## Structures for phonons 

`wfl.generate.phonopy.run()` creates displaced configs with phonopy or phono3py. 


## Normal Modes of Molecules 

Calculates normal mode directions and frequencies of molecules. From these can generate a Boltzmann sample of random displacements along multiple normal modes. See example on [Normal modes of molecules](examples.normal_modes.md). 



