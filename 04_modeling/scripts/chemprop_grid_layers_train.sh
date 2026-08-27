#!/bin/bash
#SBATCH --account=tekle_smith
#SBATCH --job-name=glyr_%a
#SBATCH --partition=burst
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=4:00:00
#SBATCH --output=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_layers/logs/job_%a.out
#SBATCH --error=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_layers/logs/job_%a.err
#SBATCH --array=0-8

# Stage-4 grid: ffn_num_layers x atom_ffn_num_layers.
# Fixed: depth=10, msg=1250, ffn=750, atom_ffn=750, dropout=0.0, lr=3e-4, bs=64.
# Submit: sbatch --array=0-8 chemprop_grid_layers_train.sh

source ~/.bashrc
conda activate chemprop_env

BASE=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling
DATA_DIR=$BASE/chemprop_data_all
GRID_DIR=$BASE/chemprop_runs/grid_layers
CONFIGS=$GRID_DIR/configs.json
mkdir -p "$GRID_DIR/logs"

IDX=$SLURM_ARRAY_TASK_ID

_cfg="import json; c=json.load(open('$CONFIGS'))[$IDX]"
DEPTH=$(       python3 -c "$_cfg; print(c['depth'])")
MSG_DIM=$(     python3 -c "$_cfg; print(c['message_hidden_dim'])")
MAX_LR=$(      python3 -c "$_cfg; print(c['max_lr'])")
BATCH_SIZE=$(  python3 -c "$_cfg; print(c['batch_size'])")
FFN_DIM=$(     python3 -c "$_cfg; print(c['ffn_hidden_dim'])")
AFFN_DIM=$(    python3 -c "$_cfg; print(c['atom_ffn_hidden_dim'])")
FFN_LAYERS=$(  python3 -c "$_cfg; print(c['ffn_num_layers'])")
AFFN_LAYERS=$( python3 -c "$_cfg; print(c['atom_ffn_num_layers'])")
DROPOUT=$(     python3 -c "$_cfg; print(c['dropout'])")
RUN_NAME=$(    python3 -c "$_cfg; print(c['run_name'])")
EPOCHS=$(      python3 -c "$_cfg; print(c['epochs'])")
PATIENCE=$(    python3 -c "$_cfg; print(c['patience'])")

OUTDIR=$GRID_DIR/$RUN_NAME
mkdir -p "$OUTDIR"

echo "=== Layers job $IDX: ffn_layers=$FFN_LAYERS  atom_ffn_layers=$AFFN_LAYERS ==="
echo "Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started: $(date)"

chemprop train \
  -i "$DATA_DIR/all_radicals.csv" \
  -s smiles \
  --atom-target-columns atom_y \
  --atom-features-path 0 "$DATA_DIR/all_radicals_atom_features.npz" \
  --splits-column split \
  -o "$OUTDIR" \
  --depth $DEPTH \
  --message-hidden-dim $MSG_DIM \
  --ffn-hidden-dim $FFN_DIM \
  --ffn-num-layers $FFN_LAYERS \
  --atom-ffn-hidden-dim $AFFN_DIM \
  --atom-ffn-num-layers $AFFN_LAYERS \
  --dropout $DROPOUT \
  --ensemble-size 1 \
  --epochs $EPOCHS \
  --patience $PATIENCE \
  --max-lr $MAX_LR \
  --batch-size $BATCH_SIZE \
  --metrics mae rmse \
  --accelerator gpu --devices 1

TRAIN_EXIT=$?
echo "Training done (exit $TRAIN_EXIT): $(date)"
[ $TRAIN_EXIT -ne 0 ] && { echo "TRAIN FAILED"; exit $TRAIN_EXIT; }

PRED_CSV=$OUTDIR/val_preds.csv
chemprop predict \
  -i "$DATA_DIR/grid_val.csv" \
  -s smiles \
  --atom-features-path 0 "$DATA_DIR/grid_val_features.npz" \
  --model-path "$OUTDIR/model_0/best.pt" \
  -o "$PRED_CSV"

[ $? -ne 0 ] && { echo "PREDICT FAILED"; exit 1; }

python3 - <<PYEOF
import ast, json, math
import numpy as np
import pandas as pd
from pathlib import Path

pred = pd.read_csv("$PRED_CSV")
val  = pd.read_csv("$DATA_DIR/grid_val.csv")
pred_col = "atom_y.1" if "atom_y.1" in pred.columns else pred.columns[-1]

def parse_list(s):
    try:    return ast.literal_eval(str(s))
    except: return [s]

y_true, y_pred = [], []
for i in range(len(val)):
    true_list = parse_list(val["atom_y"].iloc[i])
    pred_list = parse_list(pred[pred_col].iloc[i])
    rc_idx = next((j for j, v in enumerate(true_list)
                   if not math.isnan(float(v))), None)
    if rc_idx is None or rc_idx >= len(pred_list): continue
    try:
        t, p = float(true_list[rc_idx]), float(pred_list[rc_idx])
    except: continue
    if math.isnan(t) or math.isnan(p): continue
    y_true.append(math.expm1(t))
    y_pred.append(math.expm1(p))

y_true, y_pred = map(lambda x: __import__('numpy').array(x), [y_true, y_pred])
mae  = float(__import__('numpy').mean(__import__('numpy').abs(y_pred - y_true)))
rmse = float(__import__('numpy').sqrt(__import__('numpy').mean((y_pred - y_true)**2)))

result = {
    "depth": int("$DEPTH"), "message_hidden_dim": int("$MSG_DIM"),
    "ffn_hidden_dim": int("$FFN_DIM"), "atom_ffn_hidden_dim": int("$AFFN_DIM"),
    "ffn_num_layers": int("$FFN_LAYERS"), "atom_ffn_num_layers": int("$AFFN_LAYERS"),
    "dropout": float("$DROPOUT"), "max_lr": float("$MAX_LR"), "batch_size": int("$BATCH_SIZE"),
    "run_name": "$RUN_NAME", "val_mae_ev": round(mae, 6), "val_rmse_ev": round(rmse, 6),
    "n_val": len(y_true),
}
with open(Path("$OUTDIR") / "result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nRESULT: fl={result['ffn_num_layers']}  afl={result['atom_ffn_num_layers']}  "
      f"val_MAE={mae:.4f} eV  n={len(y_true)}")
PYEOF

echo "Job $IDX done: $(date)"
