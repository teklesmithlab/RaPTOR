#!/bin/bash
#SBATCH --account=tekle_smith
#SBATCH --job-name=cg_%a
#SBATCH --partition=burst
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=6:00:00
#SBATCH --output=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_search/logs/job_%a.out
#SBATCH --error=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_search/logs/job_%a.err
#SBATCH --array=0-47

# Submit: sbatch --array=0-47 chemprop_grid_search_train.sh
# (or a subset: --array=0-11 to run first 12)

source ~/.bashrc
conda activate chemprop_env

BASE=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling
DATA_DIR=$BASE/chemprop_data_all
GRID_DIR=$BASE/chemprop_runs/grid_search
CONFIGS=$GRID_DIR/configs.json
LOG_DIR=$GRID_DIR/logs
mkdir -p "$LOG_DIR"

IDX=$SLURM_ARRAY_TASK_ID

# ── Read config for this array index ────────────────────────────────────────
_cfg="import json; c=json.load(open('$CONFIGS'))[$IDX]"
DEPTH=$(    python3 -c "$_cfg; print(c['depth'])")
MSG_DIM=$(  python3 -c "$_cfg; print(c['message_hidden_dim'])")
RUN_NAME=$( python3 -c "$_cfg; print(c['run_name'])")
EPOCHS=$(   python3 -c "$_cfg; print(c['epochs'])")
PATIENCE=$( python3 -c "$_cfg; print(c['patience'])")

OUTDIR=$GRID_DIR/$RUN_NAME
mkdir -p "$OUTDIR"

echo "=== Grid job $IDX: depth=$DEPTH  message_hidden_dim=$MSG_DIM ==="
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
  --metrics mae rmse \
  --accelerator gpu --devices 1

TRAIN_EXIT=$?
echo "Training done (exit $TRAIN_EXIT): $(date)"
[ $TRAIN_EXIT -ne 0 ] && { echo "TRAIN FAILED"; exit $TRAIN_EXIT; }

# ── Evaluate on val set (eV scale) ───────────────────────────────────────────
PRED_CSV=$OUTDIR/val_preds.csv

chemprop predict \
  -i "$DATA_DIR/grid_val.csv" \
  -s smiles \
  --atom-features-path 0 "$DATA_DIR/grid_val_features.npz" \
  --model-path "$OUTDIR/model_0/best.pt" \
  -o "$PRED_CSV"

PRED_EXIT=$?
[ $PRED_EXIT -ne 0 ] && { echo "PREDICT FAILED (exit $PRED_EXIT)"; exit $PRED_EXIT; }

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

    # Find radical center index: the only non-NaN position in ground truth.
    # Do NOT use the same logic on predictions — chemprop predicts all atoms,
    # so atom 0 would always be returned instead of the radical center.
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
    "run_name":           "$RUN_NAME",
    "val_mae_ev":         round(mae,  6),
    "val_rmse_ev":        round(rmse, 6),
    "n_val":              len(y_true),
}

result_file = outdir / "result.json"
with open(result_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nRESULT: depth={result['depth']}  msg_dim={result['message_hidden_dim']}  "
      f"val_MAE={mae:.4f} eV  val_RMSE={rmse:.4f} eV  n={len(y_true)}")
PYEOF

echo "Job $IDX done: $(date)"
