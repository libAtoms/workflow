<img src="docs/wf_logo_final.png" width=200>

# Overview

Workflow is a Python toolkit for building interatomic potential creation and atomistic simulation workflows. 

The main functions of Workflow is to efficiently parallelise operations over a set of atomic configurations (Atomic Simulation Environment's "Atoms" objects). Given an operation that is defined to act on a single configuration (e.g. evaluate energy of a structure with CASTEP ASE calculator), Workflow may apply the operation to multiple configurations in parallel. Workflow also interfaces with [ExPyRe](https://github.com/libAtoms/ExPyRe/tree/main/expyre) to manage evaluation of (autoparallelized) Python functions via a queueing system on a (remote) cluster. 

For examples and more information see [documentation](https://libatoms.github.io/workflow/)


# Recent changes

v0.2.0:

- Replace all (hopefully) uses of `np.random.<sampling_method>` with passing of explicit `np.random.Generator` objects, to improve reproducibilty of scripts, and reduce chances that existing jobs will not be cached due to uncontrolled changes in random seed.  Note that this change breaks backward compatibility because many functions now _require_ an `rng` argument.

v0.1.0:

- make it possible to fire off several remote autoparallellized ops without waiting for their jobs to finish
- multi-pass calclation in `Vasp`, to allow for things like GGA followed by HSE
- MACE fitting, including remote jobs
- various bug fixes
