"""
Stage-2 grid: max_lr × batch_size with fixed architecture.

Architecture fixed at Stage-1 best: depth=10, message_hidden_dim=1250.
Produces 12 configs: 3 LR values × 4 batch sizes.
"""
import json
from pathlib import Path

BASE     = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling")
GRID_DIR = BASE / "chemprop_runs" / "grid_lr_bs"
GRID_DIR.mkdir(parents=True, exist_ok=True)

DEPTH   = 10
MSG_DIM = 1250

max_lrs     = [1e-4, 3e-4, 1e-3]
batch_sizes = [32, 64, 128, 256]

LR_TAGS = {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}

configs = []
for lr in max_lrs:
    for bs in batch_sizes:
        lr_tag   = LR_TAGS[lr]
        run_name = f"grid_lr{lr_tag}_bs{bs:03d}"
        configs.append({
            "depth":              DEPTH,
            "message_hidden_dim": MSG_DIM,
            "max_lr":             lr,
            "batch_size":         bs,
            "run_name":           run_name,
            "epochs":             50,
            "patience":           20,
        })

out_path = GRID_DIR / "configs.json"
with open(out_path, "w") as f:
    json.dump(configs, f, indent=2)

(GRID_DIR / "logs").mkdir(exist_ok=True)

print(f"Wrote {len(configs)} configs → {out_path}")
for i, c in enumerate(configs):
    print(f"  [{i:2d}] {c['run_name']}  max_lr={c['max_lr']:.1e}  batch_size={c['batch_size']}")
