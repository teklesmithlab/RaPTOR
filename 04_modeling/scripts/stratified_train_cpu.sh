#!/bin/bash
#SBATCH --account=tekle_smith
#SBATCH --job-name=stratified
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/stratified_80_10_10.out
#SBATCH --error=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/stratified_80_10_10.err

# Primary stratified 80/10/10 split (chemprop_all_prep.py's cluster_stratified_split
# on butina_cluster_assignment -- each cluster's molecules spread proportionally
# across train/val/test). Direct counterpart to cluster_grouped_train_cpu.sh:
# same architecture, same data source, same clustering column -- differs in
# exactly one variable (stratified vs grouped splitting) for the
# generalization-gap comparison. Same fixed architecture as the paper's
# production model (chemprop_runs/grid_ffn/grid_ffn750_affn750_do0p0):
# depth=10, message_hidden_dim=1250, ffn_hidden_dim=750, atom_ffn_hidden_dim=750,
# ffn_num_layers=2, atom_ffn_num_layers=2, dropout=0.0, batch_size=64,
# max_lr=0.0003, epochs=50, patience=20. CPU because burst (GPU) was too
# contended for jobs to actually start.

source ~/.bashrc
conda activate chemprop_env

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

DATA_DIR=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_data_all
OUT_DIR=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/stratified_80_10_10
mkdir -p "${OUT_DIR}"

chemprop train \
  -i "${DATA_DIR}/all_radicals.csv" \
  -s smiles \
  --atom-target-columns atom_y \
  --atom-features-path 0 "${DATA_DIR}/all_radicals_atom_features.npz" \
  --splits-column split \
  -o "${OUT_DIR}" \
  --depth 10 \
  --message-hidden-dim 1250 \
  --ffn-hidden-dim 750 \
  --ffn-num-layers 2 \
  --atom-ffn-hidden-dim 750 \
  --atom-ffn-num-layers 2 \
  --dropout 0.0 \
  --ensemble-size 1 \
  --epochs 50 \
  --patience 20 \
  --max-lr 0.0003 \
  --batch-size 64 \
  --metrics mae rmse \
  --accelerator cpu

chemprop predict \
  -i "${DATA_DIR}/all_radicals.csv" \
  -s smiles \
  --atom-features-path 0 "${DATA_DIR}/all_radicals_atom_features.npz" \
  --model-path "${OUT_DIR}/model_0/best.pt" \
  -o "${OUT_DIR}/all_preds.csv"

echo "stratified 80/10/10 run complete."
