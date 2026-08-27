"""
Collect Stage-4 (ffn_num_layers × atom_ffn_num_layers) grid results.
"""
import json
from pathlib import Path
import pandas as pd

GRID_DIR = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_runs/grid_layers")

results = []
for f in sorted(GRID_DIR.glob("grid_fl*/result.json")):
    results.append(json.load(open(f)))

config_file = GRID_DIR / "configs.json"
missing = []
if config_file.exists():
    configs = json.load(open(config_file))
    for cfg in configs:
        if not any(r["run_name"] == cfg["run_name"] for r in results):
            missing.append(cfg["run_name"])

if not results:
    print("No results yet.")
    raise SystemExit(1)

df = pd.DataFrame(results).sort_values(["ffn_num_layers", "atom_ffn_num_layers"])
df.to_csv(GRID_DIR / "grid_layers_results.csv", index=False)
print(f"Wrote {len(df)} results")

pd.set_option("display.float_format", "{:.4f}".format)
pivot = df.pivot(index="ffn_num_layers", columns="atom_ffn_num_layers", values="val_mae_ev")
print("\nVal MAE (eV) — ffn_num_layers (rows) × atom_ffn_num_layers (cols)")
print("=" * 50)
print(pivot.to_string())

best = df.loc[df["val_mae_ev"].idxmin()]
print(f"\nBest: ffn_num_layers={int(best['ffn_num_layers'])}  atom_ffn_num_layers={int(best['atom_ffn_num_layers'])}")
print(f"  val MAE  = {best['val_mae_ev']:.4f} eV")
print(f"  val RMSE = {best['val_rmse_ev']:.4f} eV")

if missing:
    print(f"\nMissing ({len(missing)}/{len(configs)}):")
    for m in missing: print(f"  {m}")
