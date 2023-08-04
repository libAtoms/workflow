<img src="wf_logo_final.png" width=200>

# Overview

Workflow is a Python toolkit for building interatomic potential creation and atomistic simulation workflows. 

The main functions of Workflow is to efficiently parallelise operations over a set of atomic configurations (Atomic Simulation Environment's "Atoms" objects). Given an operation that is defined to act on a single configuration (e.g. evaluate energy of a structure with CASTEP ASE calculator), Workflow may apply the operation to multiple configurations in parallel. Workflow also interfaces with [ExPyRe](https://github.com/libAtoms/ExPyRe/tree/main/expyre) to manage evaluation of (autoparallelized) Python functions via a queueing system on a (remote) cluster. 

For examples and more information see [documentation](https://libatoms.github.io/workflow/)


# Recent changes

Renames:

- `generic.run()` -> `generic.calculate()`
- `wfl.map.run()` -> `wfl.map.map()` 
- `wfl.generate.md.sample()` -> `wfl.generate.md.md()`
- `wfl.generate.optimize.run()` -> `wfl.generate.optimize.optimize()`
- `wfl.generate.buildcell.run()` -> `wfl.generate.buildcell.buildcell()`
- `wfl.generate.minimahopping.run()` -> `wfl.generate.minimahopping.minimahopping()`
- `phonopy.run()` -> `phonopy.phonopy()`
- `smiles.run()` -> `smiles.smiles()`
- `wfl.descriptors.quippy.calc()` -> `wfl.descriptors.quippy.calculate()`


