#!/bin/bash

#SBATCH --account=tekle_smith
#SBATCH --job-name=func  # job name
#SBATCH --ntasks-per-node=1  # number of cores requested by ORCA
#SBATCH --cpus-per-task=2   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH --time=0-11:59   # walltime in D-HH:MM
#SBATCH --output=rdkit_%j.out
#SBATCH --error=rdkit_%j.err

module purge
module load anaconda/2023.09
eval "$(conda shell.bash hook)"
conda activate ~/envs/chem

python functions.py > functions.out

