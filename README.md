<img src="docs/wf_logo_final.png" width=200>

# Overview

Workflow is a Python toolkit for building interatomic potential creation and atomistic simulation workflows. 

The main functions of Workflow is to efficiently parallelise operations over a set of atomic configurations (Atomic Simulation Environment's "Atoms" objects). Given an operation that is defined to act on a single configuration (e.g. evaluate energy of a structure with CASTEP ASE calculator), Workflow may apply the operation to multiple configurations in parallel. Workflow also interfaces with [ExPyRe](https://github.com/libAtoms/ExPyRe/tree/main/expyre) to manage evaluation of (autoparallelized) Python functions via a queueing system on a (remote) cluster. 

For examples and more information see [documentation](https://libatoms.github.io/workflow/)

`wfl` and its dependencies may be installed via `pip install wfl`. 


# Recent changes

v0.3.0:

- Update the file-based calculators (Orca, FHI-Aims, Vasp, Quantum Espresso, Castep) to work 
  with with ASE v3.23. This update breaks backwards-compatibility. For compatibility with with 
  the ASE v3.22 see use wfl v0.2.7 or earlier. 

v0.2.7:

- Latest version compatible with ASE v3.22.x. To install, use `pip install wfl==0.2.7`. 

v0.2.3:

- Add wfl.generate.neb, with required improved support for passing ConfigSet.groups() to 
  autoaparallelized functions

- Improved handling of old and new style ase.calculators.espresso.Espresso initialization

v0.2.2:

- Improve checking of DFT calculator convergence

v0.2.1:

- Fix group iterator

v0.2.0:

- Change all wfl operations to use explicit random number generator [pull 285](https://github.com/libAtoms/workflow/pull/285), to improve reproducibility of scripts and reduce the chances that on script rerun, cached jobs will not be recognized due to uncontrolled change in random seed (as in [issue 283](https://github.com/libAtoms/workflow/issues/283) and [issue 284](https://github.com/libAtoms/workflow/issues/284)).  Note that this change breaks backward compatibility because many functions now _require_ an `rng` argument, for example
  ```python
  rng = np.random.default_rng(1)
  md_configs = md.md(..., rng=rng, ...)
  ```

v0.1.0:

- make it possible to fire off several remote autoparallelized ops without waiting for their jobs to finish
- multi-pass calculation in `Vasp`, to allow for things like GGA followed by HSE
- MACE fitting, including remote jobs
- various bug fixes
