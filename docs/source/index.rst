.. workflow documentation master file, created by
   sphinx-quickstart on Tue May  4 16:26:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

########################################
Welcome to Workflow's documentation!
########################################

A Python toolkit for building interatomic potential creation and atomistic simulation workflows. 

The main functions of Workflow is to efficiently parallelise operations over a set of atomic configurations (Atomic Simulation Environment's "Atoms" objects). Given an operation that is defined to act a single configuration (e.g. evaluate energy of a structure with CASTEP ASE calculator), Workflow may apply the operation to multiple configurations in parallel. Workflow also interfaces with [ExPyRe](https://github.com/libAtoms/ExPyRe/tree/main/expyre) to manage evaluation of (autoparallelized) python functions via a queueing system on a (remote) cluster. 

Basic use of the workflow is introduced through an :ref:`example <first_example>`.  The overall design of the workflow-specific code structures (e.g. how configurations are handled, mechanism to parallelise operations) are covered in :ref:`Overview <overview>`. Currently implemented self-contained per-configuration operations are sketched out in :ref:`Operations <operations>`. Descriptions of Workflows, built out of these modular operations are described in :ref:`Workflows <workflows>`. There are also descriptions :ref:`command line interface <command_line>`, :ref:`examples <examples>` of common tasks to get started with and a :ref:`Python API <api>`. 

***************************************
Installation
***************************************

Quick start that installs all of the mandatory dependencies:

.. code-block:: sh

	pip install git+https://github.com/libAtoms/workflow

***************************************
Repository
***************************************

Please find the code, raise issues and cotribute at https://github.com/libAtoms/workflow.  


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    First example <first_example.md>
    Overview <overview.rst>
    Operations <operations.rst>
    Workflows <workflows.rst>
    Examples <examples.rst>
    Modules <modules.rst>

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
