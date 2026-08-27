#!/bin/bash
#SBATCH --account=tekle_smith
#SBATCH --job-name=loeo
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=11:55:00
#SBATCH --array=0-5
#SBATCH --output=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/loeo/%x_%a.out
#SBATCH --error=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/loeo/%x_%a.err

# Leave-one-element-out + size-matched carbon control.
#
# CPU on `short` rather than GPU: the 12h short-partition limit is ample here
# and GPU queue waits dominate actual runtime for jobs this size. Memory is
# 2G/cpu = 32G total; measured chemprop peak on this project is 4.4 GB, so
# larger requests only delay scheduling.
#
# Reduced architecture (depth 5, msg 400, ffn 300) rather than production
# (depth 10, msg 1250, ffn 750). Production measures ~3:42/epoch on CPU at
# this data size; message-passing cost scales roughly as depth x hidden^2, so
# this is ~10% of that, i.e. well under an hour for 50 epochs. The comparison
# between folds is what matters, and every fold uses the identical config --
# absolute error will sit above the production number by construction.

# NB: do not use `set -u` here -- /etc/bashrc reads BASHRCSOURCED before
# setting it, so `source ~/.bashrc` dies instantly under nounset. An earlier
# submission lost all six folds in 2 seconds that way.
source ~/.bashrc
conda activate chemprop_env
set -eo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

FOLDS=(baseline sizematch_carbon drop_carbon drop_nitrogen drop_oxygen drop_sulfur)
FOLD=${FOLDS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_data_loeo/$FOLD
OUT_DIR=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/loeo/$FOLD

echo "fold=$FOLD  node=$SLURMD_NODENAME  cpus=$SLURM_CPUS_PER_TASK  start=$(date)"
mkdir -p "$OUT_DIR"

chemprop train \
  -i "$DATA_DIR/all_radicals.csv" \
  -s smiles \
  --atom-target-columns atom_y \
  --atom-features-path 0 "$DATA_DIR/all_radicals_atom_features.npz" \
  --splits-column split \
  -o "$OUT_DIR" \
  --task-type regression \
  --depth 5 \
  --message-hidden-dim 400 \
  --ffn-hidden-dim 300 \
  --ffn-num-layers 2 \
  --atom-ffn-hidden-dim 300 \
  --atom-ffn-num-layers 2 \
  --dropout 0.0 \
  --batch-size 64 \
  --max-lr 0.0003 \
  --epochs 50 \
  --patience 20 \
  --ensemble-size 1 \
  --aggregation norm \
  --aggregation-norm 100 \
  --multi-hot-atom-featurizer-mode V2 \
  --metrics mae rmse \
  --num-workers 0 \
  --accelerator cpu

echo "fold=$FOLD done=$(date)"
