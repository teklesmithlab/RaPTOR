"""
Generate grid search configs and pre-extract val data.

Grid: 8 depths × 6 message_hidden_dims = 48 configs.
The depth × width interaction is the most important unknown after the manual
depth sweep (which used default width=300); this design maps the full surface.

Fixed at best-known values from manual hpopt:
  ffn_hidden_dim=2200, ffn_num_layers=2, atom_ffn_hidden_dim=1500,
  atom_ffn_num_layers=2, dropout=0.0

Outputs:
  chemprop_data_all/grid_val.csv        — val rows for post-training eval
  chemprop_data_all/grid_val_features.npz — corresponding atom features
  chemprop_runs/grid_search/configs.json — 48 config dicts (indexed by array job)
"""
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd

BASE = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling")
DATA_DIR = BASE / "chemprop_data_all"
RUN_DIR  = BASE / "chemprop_runs" / "grid_search"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── Grid axes ───────────────────────────────────────────────────────────────
DEPTHS    = [8, 9, 10, 11, 12, 13, 14, 15]
MSG_DIMS  = [300, 500, 750, 1000, 1250, 1500]

# Fixed hyperparameters (best from manual hpopt run)
FIXED = dict(
    ffn_hidden_dim        = 2200,
    ffn_num_layers        = 2,
    atom_ffn_hidden_dim   = 1500,
    atom_ffn_num_layers   = 2,
    dropout               = 0.0,
    ensemble_size         = 1,   # single model per trial; retrain best as 5-ensemble
    epochs                = 50,
    patience              = 20,  # early stopping — caps slow configs
)

configs = []
for depth, msg_dim in itertools.product(DEPTHS, MSG_DIMS):
    configs.append({
        "depth": depth,
        "message_hidden_dim": msg_dim,
        **FIXED,
        "run_name": f"grid_d{depth}_m{msg_dim}",
    })

out_path = RUN_DIR / "configs.json"
with open(out_path, "w") as f:
    json.dump(configs, f, indent=2)
print(f"Wrote {len(configs)} configs → {out_path}")

# ── Extract val data ─────────────────────────────────────────────────────────
print("Extracting val data...")
df = pd.read_csv(DATA_DIR / "all_radicals.csv")
val_mask   = df["split"] == "val"
val_df     = df[val_mask].reset_index(drop=True)
val_indices = val_mask.to_numpy().nonzero()[0]

val_csv = DATA_DIR / "grid_val.csv"
val_df.to_csv(val_csv, index=False)
print(f"Val CSV: {len(val_df)} rows → {val_csv}")

feat = np.load(DATA_DIR / "all_radicals_atom_features.npz")
val_arrays = [feat[f"arr_{i}"] for i in val_indices]
val_npz = DATA_DIR / "grid_val_features.npz"
np.savez(val_npz, *val_arrays)
print(f"Val features: {len(val_arrays)} arrays → {val_npz}")

print("\nReady to submit:")
print(f"  sbatch --array=0-{len(configs)-1} chemprop_grid_search_train.sh")
