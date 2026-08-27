"""
Stage-4 grid: ffn_num_layers × atom_ffn_num_layers.

Fixed at best from Stages 1-3:
  depth=10, msg=1250, ffn=750, atom_ffn=750, dropout=0.0, lr=3e-4, bs=64.

9 configs: 3 ffn_num_layers × 3 atom_ffn_num_layers.
"""
import json
from pathlib import Path

BASE     = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling")
GRID_DIR = BASE / "chemprop_runs" / "grid_layers"
GRID_DIR.mkdir(parents=True, exist_ok=True)
(GRID_DIR / "logs").mkdir(exist_ok=True)

ffn_num_layers_vals      = [1, 2, 3]
atom_ffn_num_layers_vals = [1, 2, 3]

configs = []
for fl in ffn_num_layers_vals:
    for afl in atom_ffn_num_layers_vals:
        run_name = f"grid_fl{fl}_afl{afl}"
        configs.append({
            "depth":               10,
            "message_hidden_dim":  1250,
            "max_lr":              3e-4,
            "batch_size":          64,
            "ffn_hidden_dim":      750,
            "atom_ffn_hidden_dim": 750,
            "ffn_num_layers":      fl,
            "atom_ffn_num_layers": afl,
            "dropout":             0.0,
            "run_name":            run_name,
            "epochs":              50,
            "patience":            20,
        })

out_path = GRID_DIR / "configs.json"
with open(out_path, "w") as f:
    json.dump(configs, f, indent=2)

print(f"Wrote {len(configs)} configs → {out_path}")
for i, c in enumerate(configs):
    print(f"  [{i}] {c['run_name']}  ffn_layers={c['ffn_num_layers']}  atom_ffn_layers={c['atom_ffn_num_layers']}")
