.. _overview:

######################################
Code structure
######################################

Most of :ref:`operations <operations>` in Workflow take in (optional) and return (mandatory) an ASE's ``Atoms`` object. These inputs and outputs are abstracted in ``ConfigSet`` and ``OutputSpec`` classes.  ``ConfigSet`` may be initialised with a list(s) of ``Atoms``, by providing one or multiple ``.xyz`` filenames or with a reference to an `ABCD database <https://github.com/libAtoms/abcd>`_. Similarly, returned configurations can be held in memory and subsequently used via ``OutputSpec.to_ConfigSet()``, written to a single or multiple files or uploaded to an ABCD database. This way, an operation may iterate over a ``ConfigSet`` and write ``Atoms`` to ``OutputSpec``, regardless of how the input configs were supplied or how or where to the output configs are going to be collected.

Given a ``ConfigSet`` and an operation acting on (or returning) a single ``Atoms`` instance, Workflow can parallelise the operation over all ``Atoms`` objects in ``ConfigSet`` (or an iterable leading to returning an ``Atoms`` object). This is achieved by wrapping the operation in a call to ``wfl.pipeline.iterable_loop``. In addition to parallelising on readily accessible cores, the operations may executed in a number of independently queued jobs on a HPC cluster with the help of `ExPyRe <https://github.com/libAtoms/ExPyRe>`_.

Some parts of Workflow (e.g. how many parallel processes to run) are controlled via environment variables. Off the top of my head:

* WFL_NUM_PYTHON_SUBPROCESSES
* WFL_MPIPOOL
* WFL_GAP_FIT_OMP_NUM_THREADS
* OMP_NUM_THREADS might have to be set to 1

.. toctree::
   :maxdepth: 1
   :caption: More details:

   overview.configset.rst
   overview.parallelisation.rst



