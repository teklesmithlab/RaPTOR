"""
Stage-3 grid: ffn_hidden_dim × atom_ffn_hidden_dim × dropout.

Architecture and training fixed at best from Stages 1 & 2:
  depth=10, message_hidden_dim=1250, max_lr=3e-4, batch_size=64.

36 configs: 4 ffn_hidden_dim × 3 atom_ffn_hidden_dim × 3 dropout.
"""
import json
from pathlib import Path

BASE     = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling")
GRID_DIR = BASE / "chemprop_runs" / "grid_ffn"
GRID_DIR.mkdir(parents=True, exist_ok=True)
(GRID_DIR / "logs").mkdir(exist_ok=True)

DEPTH   = 10
MSG_DIM = 1250
MAX_LR  = 3e-4
BATCH   = 64

ffn_hidden_dims      = [300, 750, 1500, 2200]
atom_ffn_hidden_dims = [300, 750, 1500]
dropouts             = [0.0, 0.1, 0.2]

configs = []
for ffn in ffn_hidden_dims:
    for affn in atom_ffn_hidden_dims:
        for do in dropouts:
            do_tag   = f"{do:.1f}".replace(".", "p")
            run_name = f"grid_ffn{ffn}_affn{affn}_do{do_tag}"
            configs.append({
                "depth":              DEPTH,
                "message_hidden_dim": MSG_DIM,
                "max_lr":             MAX_LR,
                "batch_size":         BATCH,
                "ffn_hidden_dim":     ffn,
                "atom_ffn_hidden_dim": affn,
                "ffn_num_layers":     2,
                "atom_ffn_num_layers": 2,
                "dropout":            do,
                "run_name":           run_name,
                "epochs":             50,
                "patience":           20,
            })

out_path = GRID_DIR / "configs.json"
with open(out_path, "w") as f:
    json.dump(configs, f, indent=2)

print(f"Wrote {len(configs)} configs → {out_path}")
for i, c in enumerate(configs):
    print(f"  [{i:2d}] {c['run_name']}")
