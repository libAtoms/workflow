import atexit
import os
import sys
import traceback

wfl_mpipool = None


def shutdown_and_barrier(pool, comm):
    print(comm.rank, 'MPIPOOL: shutdown_and_barrier calling barrier')
    pool.shutdown()
    comm.Barrier()


def init(verbose=1):
    """Startup code when mpipool will be used. Only master MPI task exists
    function, others wait to do mpipool stuff.  Initialises
    `mpipool_support.wfl_mpipool` with created `MPIExecutor` object.

    Parameters
    ----------
    verbose: int, default 1
        * >= 1 : minimal start/end messages
        * > 1  : print stack trace at startup, to tell where it was called from
    """
    global wfl_mpipool

    if wfl_mpipool is None and 'WFL_MPIPOOL' in os.environ:
        # check version
        import mpipool
        from packaging.version import parse
        assert parse(mpipool.__version__) >= parse('1.0.0')

        # wfl_mpipool is not defined, must be first time
        if verbose > 0:
            from mpi4py import MPI
        from mpipool import MPIExecutor

        if verbose > 0:
            print(MPI.COMM_WORLD.rank, "MPIPOOL: wfl creating pool")
        if verbose > 1:
            for item in traceback.format_stack():
                print(MPI.COMM_WORLD.rank, "MPIPOOL: STACK", item, end='')
        wfl_mpipool = MPIExecutor()

        if wfl_mpipool.is_worker():
            print(MPI.COMM_WORLD.rank, 'MPIPOOL: worker calling barrier')
            MPI.COMM_WORLD.Barrier()
            if "pytest" in sys.modules:
                return
            else:
                exit()

        atexit.register(shutdown_and_barrier, wfl_mpipool, MPI.COMM_WORLD)

        if verbose > 0:
            print(MPI.COMM_WORLD.rank, "MPIPOOL: wfl continuing after creating pool")
