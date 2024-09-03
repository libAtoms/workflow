<img src="docs/wf_logo_final.png" width=200>

# Overview

Workflow is a Python toolkit for building interatomic potential creation and atomistic simulation workflows. 

The main functions of Workflow is to efficiently parallelise operations over a set of atomic configurations (Atomic Simulation Environment's "Atoms" objects). Given an operation that is defined to act on a single configuration (e.g. evaluate energy of a structure with CASTEP ASE calculator), Workflow may apply the operation to multiple configurations in parallel. Workflow also interfaces with [ExPyRe](https://github.com/libAtoms/ExPyRe/tree/main/expyre) to manage evaluation of (autoparallelized) Python functions via a queueing system on a (remote) cluster. 

For examples and more information see [documentation](https://libatoms.github.io/workflow/)

`wfl` and its dependencies may be installed via `pip install wfl`. 


# Recent changes

v0.3.1:

- additional updates to file-based calculators for ASE v3.23.
- fixes to parity plots

v0.3.0:

- Update the file-based calculators (Orca, FHI-Aims, Vasp, Quantum Espresso, Castep) to work 
  with with ASE v3.23. This update breaks backwards-compatibility. For compatibility with with 
  the ASE v3.22 see use wfl v0.2.8 or earlier. 

v0.2.8:

- Latest version compatible with ASE v3.22.x. To install, use `pip install wfl==0.2.8`. 

For older changes see [documentation](https://libatoms.github.io/workflow).

