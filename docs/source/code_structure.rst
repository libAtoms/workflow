.. _code_structure:

######################################
Code structure
######################################

Most of :ref:`operations <operations>` in Workflow take in (optional) and return (mandatory) an ASE's ``Atoms`` object. These inputs and outputs are abstracted in ``ConfigSet_in`` and ``ConfigSet_out`` classes.  ``ConfigSet_in`` may be initialised with a list(s) of ``Atoms``, by providing one or multiple ``.xyz`` filenames or with a reference to an `ABCD database <https://github.com/libAtoms/abcd>`_. Similarly, returned configurations can be held in memory and subsequently used via ``ConfigSet_out.to_ConfigSet_in()``, written to a single or multiple files or uploaded to an ABCD database. This way, an operation may iterate over a ``ConfigSet_in`` and write ``Atoms`` to ``ConfigSet_out``, regardless of how the input configs were supplied or how or where to the output configs are going to be collected.

Given a ``ConfigSet_in`` and an operation acting on (or returning) a single ``Atoms`` instance, Workflow can parallelise the operation over all ``Atoms`` objects in ``ConfigSet_in`` (or an iterable leading to returning an ``Atoms`` object). This is achieved by wrapping the operation in a call to ``wfl.pipeline.iterable_loop``. In addition to parallelising on readily accessible cores, the operations may executed in a number of independently queued jobs on a HPC cluster with the help of `ExPyRe <https://github.com/libAtoms/ExPyRe>`_. 

Some parts of Workflow (e.g. how many parallel processes to run) are controlled via environment variables. Off the top of my head: 

* WFL_NUM_PYTHON_SUBPROCESSES
* WFL_MPIPOOL
* GAP_FIT_OMP_NUM_THREADS 
* OMP_NUM_THREADS might have to be set to 1  

.. toctree::
   :maxdepth: 1
   :caption: More details:

   code_structure.configset.rst
   code_structure.parallelisation.rst



