#!/bin/bash
#SBATCH --account=tekle_smith
#SBATCH --job-name=kcurve
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=11:55:00
#SBATCH --array=0-26%9
#SBATCH --output=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/kcurve/%x_%a.out
#SBATCH --error=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/kcurve/%x_%a.err

# End-to-end learning curve for adding a NEW element.
#
# Each fold trains on every non-E molecule plus k molecules of element E, and
# is tested on the untouched all-element test set. Unlike the frozen-probe
# version of this curve, the representation itself only ever sees k molecules
# of E, so this measures what the element's data is actually worth.
#
# %9 throttles to 9 concurrent tasks so the array does not monopolise `short`.
# Same reduced architecture as the LOEO folds (job 11722439) so the k=0 point
# (drop_E) and k=all are directly comparable.
#
# NB: do not use `set -u` -- /etc/bashrc reads BASHRCSOURCED before setting it,
# so `source ~/.bashrc` dies instantly under nounset.

source ~/.bashrc
conda activate chemprop_env
set -eo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

DATA_ROOT=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_data_kcurve
FOLD=$(python3 -c "import json;print(json.load(open('$DATA_ROOT/fold_names.json'))[$SLURM_ARRAY_TASK_ID])")

DATA_DIR=$DATA_ROOT/$FOLD
OUT_DIR=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/kcurve/$FOLD

echo "fold=$FOLD  node=$SLURMD_NODENAME  start=$(date)"
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
