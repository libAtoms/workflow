#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --nice=20
#SBATCH --hint=nomultithread
#SBATCH --job-name=test
#SBATCH --partition=standard
#SBATCH --time=0:01:00
#SBATCH --output=stdout
#SBATCH --error=stderr

hostname
