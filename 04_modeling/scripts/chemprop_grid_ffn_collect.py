"""
Collect Stage-3 (ffn_hidden_dim × atom_ffn_hidden_dim × dropout) grid results.

Usage:
    python chemprop_grid_ffn_collect.py

Reads result.json from each grid_ffn*/ directory.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID_DIR = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_ffn")

results = []
for result_file in sorted(GRID_DIR.glob("grid_ffn*/result.json")):
    with open(result_file) as f:
        results.append(json.load(f))

config_file = GRID_DIR / "configs.json"
missing = []
if config_file.exists():
    configs = json.load(open(config_file))
    for cfg in configs:
        name = cfg["run_name"]
        if not any(r["run_name"] == name for r in results):
            missing.append(name)

if not results:
    print("No result.json files found.")
    raise SystemExit(1)

df = pd.DataFrame(results).sort_values(["ffn_hidden_dim", "atom_ffn_hidden_dim", "dropout"])

out_csv = GRID_DIR / "grid_ffn_results.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote {len(df)} results → {out_csv}")

pd.set_option("display.float_format", "{:.4f}".format)

# ── Pivot: ffn_hidden_dim × atom_ffn_hidden_dim (best dropout per cell) ───────
best_per_cell = df.groupby(["ffn_hidden_dim", "atom_ffn_hidden_dim"])["val_mae_ev"].min().unstack()
print("\nBest val MAE (eV) per cell — ffn_hidden_dim (rows) × atom_ffn_hidden_dim (cols)")
print("=" * 60)
print(best_per_cell.to_string())

# ── Dropout marginal ──────────────────────────────────────────────────────────
do_s = df.groupby("dropout")["val_mae_ev"].agg(["min", "mean"])
do_s.columns = ["best_mae", "mean_mae"]
print("\nMAE by dropout (across all ffn dims):")
print(do_s.round(4).to_string())

# ── FFN dim marginal ──────────────────────────────────────────────────────────
ffn_s = df.groupby("ffn_hidden_dim")["val_mae_ev"].agg(["min", "mean"])
ffn_s.columns = ["best_mae", "mean_mae"]
print("\nMAE by ffn_hidden_dim:")
print(ffn_s.round(4).to_string())

affn_s = df.groupby("atom_ffn_hidden_dim")["val_mae_ev"].agg(["min", "mean"])
affn_s.columns = ["best_mae", "mean_mae"]
print("\nMAE by atom_ffn_hidden_dim:")
print(affn_s.round(4).to_string())

# ── Best config ───────────────────────────────────────────────────────────────
best = df.loc[df["val_mae_ev"].idxmin()]
print(f"\nBest: ffn_hidden_dim={int(best['ffn_hidden_dim'])}  "
      f"atom_ffn_hidden_dim={int(best['atom_ffn_hidden_dim'])}  "
      f"dropout={best['dropout']:.1f}")
print(f"  val MAE  = {best['val_mae_ev']:.4f} eV")
print(f"  val RMSE = {best['val_rmse_ev']:.4f} eV")
print(f"  run dir  = {GRID_DIR / best['run_name']}")

if missing:
    print(f"\nMissing ({len(missing)}/{len(configs)} not yet done):")
    for m in missing:
        print(f"  {m}")
