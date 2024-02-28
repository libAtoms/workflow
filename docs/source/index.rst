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

	python3 -m pip install git+https://github.com/libAtoms/workflow

.. warning::

    `wfl` requires ASE, so `ase` is listed as a `pip` dependency,
    and if not already installed, `pip install` will install the latest
    `pypi` release.  However, because of the large delay in producing new
    releases, the latest `pypi` version is often quite old, and `wfl`
    has some functionality that requires a newer version.  To ensure
    a sufficiently up-to-date version is available, before installing
    `wfl` install the latest `ase` from gitlab, with a command such as

    .. code-block:: sh

            python3 -m pip install git+https://gitlab.com/ase/ase

***************************************
Repository
***************************************

Please find the code, raise issues and cotribute at https://github.com/libAtoms/workflow.  


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
