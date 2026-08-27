"""
Collect results from all completed grid search jobs into a summary CSV.

Usage:
    python chemprop_grid_search_collect.py

Reads result.json from each grid_d{depth}_m{msg_dim}/ directory.
Writes chemprop_runs/grid_search/grid_results.csv and prints a summary table.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID_DIR = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_search")

results = []
missing = []

for result_file in sorted(GRID_DIR.glob("grid_d*/result.json")):
    with open(result_file) as f:
        results.append(json.load(f))

config_file = GRID_DIR / "configs.json"
if config_file.exists():
    configs = json.load(open(config_file))
    for cfg in configs:
        name = cfg["run_name"]
        if not any(r["run_name"] == name for r in results):
            missing.append(name)

if not results:
    print("No result.json files found. Have any array jobs completed?")
    raise SystemExit(1)

df = pd.DataFrame(results).sort_values(["depth", "message_hidden_dim"])
df = df[["depth", "message_hidden_dim", "val_mae_ev", "val_rmse_ev", "n_val", "run_name"]]

out_csv = GRID_DIR / "grid_results.csv"
df.to_csv(out_csv, index=False)
print(f"Wrote {len(df)} results → {out_csv}")

# ── Pivot table ───────────────────────────────────────────────────────────────
pivot = df.pivot(index="depth", columns="message_hidden_dim", values="val_mae_ev")
print("\nVal MAE (eV) — depth (rows) × message_hidden_dim (cols)")
print("=" * 60)
pd.set_option("display.float_format", "{:.4f}".format)
print(pivot.to_string())

# ── Best config ───────────────────────────────────────────────────────────────
best = df.loc[df["val_mae_ev"].idxmin()]
print(f"\nBest config: depth={int(best['depth'])}  message_hidden_dim={int(best['message_hidden_dim'])}")
print(f"  val MAE  = {best['val_mae_ev']:.4f} eV")
print(f"  val RMSE = {best['val_rmse_ev']:.4f} eV")
print(f"  run dir  = {GRID_DIR / best['run_name']}")

if missing:
    print(f"\nMissing ({len(missing)}/{len(configs)} not yet done):")
    for m in missing:
        print(f"  {m}")

# ── Marginal summaries ────────────────────────────────────────────────────────
depth_summary = df.groupby("depth")["val_mae_ev"].agg(["min", "mean"])
depth_summary.columns = ["best_mae", "mean_mae"]
print("\nMAE by depth (across all widths):")
print(depth_summary.round(4).to_string())

width_summary = df.groupby("message_hidden_dim")["val_mae_ev"].agg(["min", "mean"])
width_summary.columns = ["best_mae", "mean_mae"]
print("\nMAE by message_hidden_dim (across all depths):")
print(width_summary.round(4).to_string())
