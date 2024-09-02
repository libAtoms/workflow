.. workflow documentation master file, created by
   sphinx-quickstart on Tue May  4 16:26:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

########################################
Welcome to Workflow's documentation!
########################################

A Python toolkit for building interatomic potential creation and atomistic simulation workflows. 

The main function of Workflow is to efficiently parallelise operations over a set of atomic configurations (Atomic Simulation Environment's "Atoms" objects). Given an operation that is defined to act a single configuration (e.g. evaluate energy of a structure with CASTEP ASE calculator), Workflow may apply the operation to multiple configurations in parallel. Workflow also interfaces with `ExPyRe <https://github.com/libAtoms/ExPyRe/tree/main/expyre>`_ to manage evaluation of (autoparallelized) python functions via a queueing system on a (remote) cluster. 

Basic use of the workflow is introduced through an :ref:`example <first_example>`.  The overall design of the workflow-specific code structures (e.g. how configurations are handled, mechanism to parallelise operations and to submit as queued jobs) are covered in :ref:`Overview <overview>`. Currently implemented self-contained per-configuration operations are sketched out in :ref:`Operations <operations>`. Description of GAP-RSS workflow, built out of these modular operations is described in :ref:`GAP-RSS <rss>`. There are also  :ref:`examples <examples>` of common tasks to get started with and a :ref:`Python API <api>`. 

***************************************
Installation
***************************************

Quick start that installs all of the mandatory dependencies:

.. code-block:: sh

	python3 -m pip install wfl


***************************************
Repository
***************************************

Please find the code, raise issues and cotribute at https://github.com/libAtoms/workflow.  


***************************************
Development
***************************************

To install all dependencies needed for running unit tests:

.. code-block:: sh

	python3 -m pip install /path/to/workflow[test] 


Some of the `wfl` functions rely on the `quippy-ase` package, which currently (2 September 2024) supports no higher Python versions than v3.9.

The file-based calculator tests need ASE calculator configiguration file to be present; `see the ASE documentation <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#calculator-configuration>`. 


***************************************
Recent Changes
***************************************

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


***************************************
Contents
***************************************


.. toctree::
    :maxdepth: 2

    First example <first_example.md>
    Overview <overview.rst>
    Operations <operations.rst>
    Examples <examples.rst>
    GAP-RSS <workflows.rss.rst>
    Command line interface <command_line.automatic_docs.rst>
    Modules <modules.rst>

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
