.. _parallelisation: 

########################################
Automatic parallelization of tasks
########################################


Much of the pipeline, including the input/output facilitated by ``ConfigSet_in``/``ConfigSet_out``, was designed that so that simple operations that need to be done to many configurations could easily be parallelized.  The mechanism for that is wrapping the fundamental operations in a call to ``wfl.pipeline.iterable_loop``.


****************************************
Runtime view
****************************************


Once a function that operates on individual configs has been wrapped, the user can get parallelization to happen in one of two different ways.  The first is the use of the env var ``WFL_AUTOPARA_NPOOL``.  This variable should be set to a number, and that many python threads will be created (using `multiprocessing.pool.Pool <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool>`_).  The script should be run as usual.

The other is to set the env var ``WFL_MPIPOOL`` to any value, and to run the script with ``mpirun`` (or whatever is appropriate for the installed MPI implementation).  All MPI tasks will be used to parallelize using `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ and `mpipool <https://github.com/mpipool/mpipool>`_.

****************************************
Programming-time view
****************************************

To code an operation that can be parallelized over configuration (or any other iterable), it needs to be implemented as a function that it takes, normally as its first argument, an iterable, e.g. a list of ``ase.atoms.Atoms``, and returns a list of ``ase.atoms.Atoms`` the same length.  Input can be any iterable, but output must be ``Atoms`` or nothing, no other objects.  If the actual work is being done in a function called ``op``, it should be defined as 

.. code-block:: python

    def op(atoms, arg1, arg2, arg3, ...):
	    output_ats = []
	    for at in atoms:
            <do some stuff with at and argN>
        output_ats.append(at)
        return output_ats

An auto-parallelized wrapper function can then be defined as

.. code-block:: python

  def wrapped_op(inputs, outputs, arg1, arg2, arg3, ...)
    return iterable_loop(iterable=inputs, configset_out=outputs, op=op, chunksize=<N>,
            arg1=arg1, arg2=arg2, arg3=arg3, ...)

``chunksize`` controls how many items from the input iterable are passed to each call of ``op()`` (to reduce startup overhead).  All arguments _must be_ pickleable.  If something that cannot be pickled must be passed (e.g. a QUIP ``Potential``), it must be passed in some way, e.g. a constructor function and its arguments, that _can_ be pickled.  For things that need to happen once per thread, e.g. random number initialization, there is an ``initializer`` argument to ``iterable_loop()``.

There are many examples of this, including the descriptor calculator, and (with initializers) md and minim. 


MPI with ``WFL_MPIPOOL``
================================

The operator-level implementation of ``mpipool``-based parallelization is the same.  The only difference is that (assuming that the script as a whole is written for a single task/thread), at startup the script has to call ``wfl.mpipool_support.init()``.  This function will hang for every task except for ``rank == 0``, and all those tasks will wait for things to be done through the ``mpipool`` mechanism.  Task 0 should continue, doing whatever it needs to, and when it calls the wrapped operation it will be parallelized over all MPI tasks.

