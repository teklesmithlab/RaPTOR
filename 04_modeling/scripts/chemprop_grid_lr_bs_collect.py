"""
Collect Stage-2 (max_lr × batch_size) grid search results.

Usage:
    python chemprop_grid_lr_bs_collect.py

Reads result.json from each grid_lr*/bs*/ directory.
Writes chemprop_runs/grid_lr_bs/grid_lr_bs_results.csv and prints pivot table.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID_DIR = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_lr_bs")

results = []
for result_file in sorted(GRID_DIR.glob("grid_lr*/result.json")):
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
    print("No result.json files found. Have any array jobs completed?")
    raise SystemExit(1)

df = pd.DataFrame(results).sort_values(["max_lr", "batch_size"])

out_csv = GRID_DIR / "grid_lr_bs_results.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote {len(df)} results → {out_csv}")

# ── Pivot: max_lr × batch_size ────────────────────────────────────────────────
pivot = df.pivot(index="max_lr", columns="batch_size", values="val_mae_ev")
print("\nVal MAE (eV) — max_lr (rows) × batch_size (cols)")
print("=" * 60)
pd.set_option("display.float_format", "{:.4f}".format)
print(pivot.to_string())

# ── Best config ───────────────────────────────────────────────────────────────
best = df.loc[df["val_mae_ev"].idxmin()]
print(f"\nBest: max_lr={best['max_lr']:.1e}  batch_size={int(best['batch_size'])}")
print(f"  val MAE  = {best['val_mae_ev']:.4f} eV")
print(f"  val RMSE = {best['val_rmse_ev']:.4f} eV")
print(f"  run dir  = {GRID_DIR / best['run_name']}")

if missing:
    print(f"\nMissing ({len(missing)}/{len(configs)} not yet done):")
    for m in missing:
        print(f"  {m}")

# ── Marginals ─────────────────────────────────────────────────────────────────
lr_s = df.groupby("max_lr")["val_mae_ev"].agg(["min", "mean"])
lr_s.columns = ["best_mae", "mean_mae"]
print("\nMAE by max_lr (across all batch sizes):")
print(lr_s.round(4).to_string())

bs_s = df.groupby("batch_size")["val_mae_ev"].agg(["min", "mean"])
bs_s.columns = ["best_mae", "mean_mae"]
print("\nMAE by batch_size (across all LRs):")
print(bs_s.round(4).to_string())
