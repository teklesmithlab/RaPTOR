#!/bin/bash
#SBATCH --account=tekle_smith
#SBATCH --job-name=glbs_%a
#SBATCH --partition=burst
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=6:00:00
#SBATCH --output=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_lr_bs/logs/job_%a.out
#SBATCH --error=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_lr_bs/logs/job_%a.err
#SBATCH --array=0-11

# Stage-2 grid: max_lr × batch_size, architecture fixed at depth=10 msg_dim=1250.
# Submit: sbatch --array=0-11%10 chemprop_grid_lr_bs_train.sh

source ~/.bashrc
conda activate chemprop_env

BASE=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling
DATA_DIR=$BASE/chemprop_data_all
GRID_DIR=$BASE/chemprop_runs/grid_lr_bs
CONFIGS=$GRID_DIR/configs.json
LOG_DIR=$GRID_DIR/logs
mkdir -p "$LOG_DIR"

IDX=$SLURM_ARRAY_TASK_ID

# ── Read config ───────────────────────────────────────────────────────────────
_cfg="import json; c=json.load(open('$CONFIGS'))[$IDX]"
DEPTH=$(      python3 -c "$_cfg; print(c['depth'])")
MSG_DIM=$(    python3 -c "$_cfg; print(c['message_hidden_dim'])")
MAX_LR=$(     python3 -c "$_cfg; print(c['max_lr'])")
BATCH_SIZE=$( python3 -c "$_cfg; print(c['batch_size'])")
RUN_NAME=$(   python3 -c "$_cfg; print(c['run_name'])")
EPOCHS=$(     python3 -c "$_cfg; print(c['epochs'])")
PATIENCE=$(   python3 -c "$_cfg; print(c['patience'])")

OUTDIR=$GRID_DIR/$RUN_NAME
mkdir -p "$OUTDIR"

echo "=== LR/BS job $IDX: max_lr=$MAX_LR  batch_size=$BATCH_SIZE ==="
echo "Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Output dir: $OUTDIR"
echo "Started: $(date)"

# ── Train ────────────────────────────────────────────────────────────────────
chemprop train \
  -i "$DATA_DIR/all_radicals.csv" \
  -s smiles \
  --atom-target-columns atom_y \
  --atom-features-path 0 "$DATA_DIR/all_radicals_atom_features.npz" \
  --splits-column split \
  -o "$OUTDIR" \
  --depth $DEPTH \
  --message-hidden-dim $MSG_DIM \
  --ffn-hidden-dim 2200 \
  --ffn-num-layers 2 \
  --atom-ffn-hidden-dim 1500 \
  --atom-ffn-num-layers 2 \
  --dropout 0.0 \
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

# ── Predict on val set ────────────────────────────────────────────────────────
PRED_CSV=$OUTDIR/val_preds.csv

chemprop predict \
  -i "$DATA_DIR/grid_val.csv" \
  -s smiles \
  --atom-features-path 0 "$DATA_DIR/grid_val_features.npz" \
  --model-path "$OUTDIR/model_0/best.pt" \
  -o "$PRED_CSV"

PRED_EXIT=$?
[ $PRED_EXIT -ne 0 ] && { echo "PREDICT FAILED (exit $PRED_EXIT)"; exit $PRED_EXIT; }

# ── Evaluate (eV scale, radical center index from ground truth) ───────────────
python3 - <<PYEOF
import ast, json, math
import numpy as np
import pandas as pd
from pathlib import Path

pred_csv = "$PRED_CSV"
val_csv  = "$DATA_DIR/grid_val.csv"
outdir   = Path("$OUTDIR")

pred = pd.read_csv(pred_csv)
val  = pd.read_csv(val_csv)

pred_col = "atom_y.1" if "atom_y.1" in pred.columns else pred.columns[-1]

def parse_list(s):
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return [s]

y_true, y_pred = [], []
for i in range(len(val)):
    true_list = parse_list(val["atom_y"].iloc[i])
    pred_list = parse_list(pred[pred_col].iloc[i])

    rc_idx = None
    for j, v in enumerate(true_list):
        try:
            f = float(v)
            if not math.isnan(f):
                rc_idx = j
                break
        except (ValueError, TypeError):
            pass

    if rc_idx is None or rc_idx >= len(pred_list):
        continue
    try:
        t = float(true_list[rc_idx])
        p = float(pred_list[rc_idx])
    except (ValueError, TypeError):
        continue
    if math.isnan(t) or math.isnan(p):
        continue

    y_true.append(np.expm1(t))
    y_pred.append(np.expm1(p))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
mae  = float(np.mean(np.abs(y_pred - y_true)))
rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

result = {
    "depth":              int("$DEPTH"),
    "message_hidden_dim": int("$MSG_DIM"),
    "max_lr":             float("$MAX_LR"),
    "batch_size":         int("$BATCH_SIZE"),
    "run_name":           "$RUN_NAME",
    "val_mae_ev":         round(mae,  6),
    "val_rmse_ev":        round(rmse, 6),
    "n_val":              len(y_true),
}

with open(outdir / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nRESULT: max_lr={result['max_lr']:.1e}  bs={result['batch_size']}  "
      f"val_MAE={mae:.4f} eV  val_RMSE={rmse:.4f} eV  n={len(y_true)}")
PYEOF

echo "Job $IDX done: $(date)"
