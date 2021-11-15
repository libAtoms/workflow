# MPI tests

These tests are intended to be run with MPI, and test the parallelism of the
code in this way.

How to run there:
- install: `mpi4py` and `mpipool` linked to your MPI installation
- run the tests with:
```bash
export WFL_MPIPOOL=2

# no coverage
mpirun -n 2 pytest --with-mpi -k mpi

# with coverage
mpirun -n 2 pytest --cov=wfl --cov-report term --cov-config=tests/.coveragerc --cov-report term-missing --cov-report term:skip-covered --with-mpi --cov-append -k mpi
```
- there will be duplicate output in the terminal window

The latter appends the coverage to any done before, which should be helpful.
The GitHub CI is set up such that this is happening automatically on the chosen version of 
python where we are doing coverage as well.

## Gotchas:
- these tests need the pytest mpi decorator
- any test in this directory will be run with the MPI (due to `-k mpi` in pytest)
- any test in this directory *not* having the mpi decorator on top of it will be run wihtout MPI 
  as well
- not all tests elsewhere are compatible with MPI
- any test elsewhere that has MPI in its name will be ran with MPI as well, be carefl