.. _parallelisation: 

########################################
Automatic parallelization of tasks
########################################


Much of the pipeline, including the input/output facilitated by ``ConfigSet``/``OutputSpec``, was designed that so that simple operations that need to be done to many configurations could easily be parallelized.  The mechanism for that is wrapping the fundamental operations in a call to ``wfl.autoparallelize.base.autoparallelize``. This parallelization can in principle include two levels

* Splitting up the input iterator into groups, each of which is processed by a separate
  python subprocess.
* Splitting up the input iterator into groups, each of which is processed in a separate
  job submitted to a local or remote queuing system. The job can then use python
  subprocess parallelization itself. [remote jobs not documented here yet]

.. warning::
    Autoparallelized operations will use cached output files.  Even if the code that is executed by
    the operation has changed, the previous and perhaps wrong output will be used.
    See warning in :doc:`overview.configset`

*****************************************************
Programming script that use parallelized operations 
*****************************************************

Parallelized operations can be called from a python script, and have

* the first function argument is the inputs, as an iterator (usually ``ConfigSet``, but some operations, e.g. ``buildcell``,
  just use a counter like ``range(n)``).
* The second function argument is outputs, as an ``OutputSpec``. There is no support for returned values of any type other
  than ``Atoms`` store in a ``ConfigSet``.

The function will return a ``ConfigSet`` containing the output configs, which will be stored wherever the ``OutputSpec``
object's constructor arguments indicated.

The optional argument ``num_inputs_per_python_subprocess`` will determine how many input values will
be procesed in each call to the low level function.  This defaults to 1, but can be increasd to reduce
overhead that happens once per call, e.g. the construction of expensive ASE calculators like ``quippy.potential.Potential``
with a GAP model.

===========================
MPI with ``WFL_MPIPOOL``
===========================

If it is necessary to parallelize over more than one node, ``mpipool``
(see below) can be used. In this case (assuming that the script as a
whole is written for a single task/thread), at startup the script has
to call ``wfl.autoparallelize.mpipool_support.init()``.  This function
will hang for every task except for ``rank == 0``, and all those tasks
will wait for things to be done through the ``mpipool`` mechanism.
Task 0 should continue, doing whatever it needs to, and when it calls
the wrapped operation it will be parallelized over all MPI tasks.


****************************************
Runtime control over parallelization
****************************************

Once a function that operates on individual configs has been wrapped,
the user can get parallelization to happen in one of two different ways.

========================================
Single node using python subprocesses
========================================

The first is using python threads,
created using `multiprocessing.pool.Pool
<https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool>`_.
The number of threads is controlled by an integer, passed in to the
function as an optional ``num_python_subprocesses`` argument, or stored
in the env var ``WFL_NUM_PYTHON_SUBPROCESSES``.  The script should be
started with a normal run of the python executable.


========================================
Multiple nodes using MPI
========================================

If using ``mpipool`` the env var ``WFL_MPIPOOL`` must be set to any value.
In this case  the script must be run with ``mpirun`` (or whatever is
appropriate for the installed MPI implementation), and the number of
python subprocesses will be determined by the number of ``mpirun``
MPI tasks (e.g. ``-np N``).  All MPI tasks will be used to parallelize
using `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ and
`mpipool <https://github.com/mpipool/mpipool>`_.


****************************************
Creating auto-parallelized functions
****************************************

To code an operation that can be parallelized over configuration (or
any other iterable), it needs to be implemented as a function that
takes, normally as its first argument, an **iterable**, e.g. a list of
``ase.atoms.Atoms``, and returns a **list** of ``ase.atoms.Atoms``
of the same length.  Input can be any iterable, but output must be
``Atoms`` or nothing, no other objects.  If the actual work is being
done in a function called ``op``, it should be defined as

.. code-block:: python

    def op(atoms, arg1, arg2, arg3, ...):
        output_ats = []
        for at in atoms:
            <do some stuff with at and argN>
            output_ats.append(at)
        return output_ats

An auto-parallelized wrapper function can then be defined as (for example)

.. code-block:: python

    def autopara_op(*args, **kwargs):
        return autoparallelize(op, *args, def_num_inputs_per_python_subprocess=10, **kwargs)
    autopara_op.__doc__ = autoparallelize_docstring(op.__foc__, "Atoms")

``def_num_inputs_per_python_subprocess`` controls how many items
(by default) from the input iterable are passed to each call of
``op()`` (to reduce startup overhead).  All arguments *must be*
pickleable.  If something that cannot be pickled must be passed (e.g. a
``quippy.potential.Potential``), it must be passed in some way, e.g. a
constructor function and its arguments, that _can_ be pickled (see ``wfl.calculators.generic``).  For things
that need to happen once per thread, e.g. random number initialization,
there is an ``initializer`` argument to ``autoparallelize()`` (see ``wfl.generate.md``).

There are many examples of this, including the descriptor calculator, and (with initializers) md and minim. 

