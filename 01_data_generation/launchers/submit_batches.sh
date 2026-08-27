#!/bin/bash

#SBATCH --account=tekle_smith
#SBATCH --job-name=
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-11:59

export NBOEXE=/insomnia001/depts/tekle_smith/users/softwares/NBO/nbo7/bin/nbo7.i8.exe
export NBO7KEY=/insomnia001/depts/tekle_smith/users/softwares/NBO/nbo7/nbo7.key

module purge
module load anaconda/2023.09
source ~/.bashrc
conda activate ~/envs/chem

python submit_batches.py --root_directory "/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation" --batch_number 2110 > "/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/submit_batches.out"
