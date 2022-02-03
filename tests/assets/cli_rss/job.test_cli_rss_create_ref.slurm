#!/bin/bash
#SBATCH --job-name=test_cli_rss_create_ref
#SBATCH --partition=CPU_16
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --output=test_cli_rss_create_ref.stdout
#SBATCH --error=test_cli_rss_create_ref.stderr
#SBATCH --time=24:00:00

pwd

rm -rf test_cli_rss_create_ref
mkdir test_cli_rss_create_ref
export GAP_RSS_TEST_SETUP=$PWD/test_cli_rss_create_ref

module load vasp
# make sure it's not OpenMP enabled, so gap_fit is deterministic
module unload quip_variant quip
module load quip

export VASP_COMMAND=vasp.serial
export VASP_COMMAND_GAMMA=vasp.gamma_serial
export VASP_PP_PATH=$VASP_PATH/pot/rev_54/PBE
export GRIF_BUILDCELL_CMD=$HOME/src/work/AIRSS/airss-0.9.1/src/buildcell/src/buildcell

export WFL_AUTOPARA_NPOOL=16

pytest --runslow -rxXs -s tests/test_cli_rss.py 

cd $PBS_O_WORKDIR
cp job.test_cli_rss_create_ref.slurm test_cli_rss_create_ref.stdout tests/assets/cli_rss
