#!/bin/bash
#SBATCH --job-name=test_cli_rss_create_ref
#SBATCH --partition=CPU_16
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --output=test_cli_rss_create_ref.stdout
#SBATCH --error=test_cli_rss_create_ref.stderr
#SBATCH --time=6:00:00

pwd

rm -rf test_cli_rss_create_ref
mkdir test_cli_rss_create_ref
export GAP_RSS_TEST_SETUP=$PWD/test_cli_rss_create_ref

module purge
module load compiler/gnu python/system python_extras/quippy dft/vasp
# make sure it's not OpenMP enabled, so gap_fit is deterministic
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export ASE_VASP_COMMAND=vasp.serial
export ASE_VASP_COMMAND_GAMMA=vasp.gamma.serial
export VASP_PP_PATH=$VASP_PATH/pot/rev_54/PBE
export GRIF_BUILDCELL_CMD=$HOME/src/work/AIRSS/airss-0.9.1/src/buildcell/src/buildcell

export WFL_NUM_PYTHON_SUBPROCESSES=16

pytest --runslow -rxXs -s tests/test_cli_rss.py 

cp job.test_cli_rss_create_ref.slurm test_cli_rss_create_ref.stdout tests/assets/cli_rss
