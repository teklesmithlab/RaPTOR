#!/bin/bash
#SBATCH --account=tekle_smith
#SBATCH --job-name=raptor_retrain
#SBATCH --partition=tekle_smith1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=8:00:00
#SBATCH --output=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_retrain_80_10_10.out
#SBATCH --error=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_retrain_80_10_10.err

# Trial_023 config, no BN, 80/10/10 cluster-based split.
# BN was tested and found to worsen CCSD(T) performance (0.1024 vs 0.0922 eV).

source ~/.bashrc
conda activate chemprop_env

BASE=/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling
DATA_DIR=$BASE/chemprop_data_all
OUTDIR=$BASE/chemprop_runs/raptor_80_10_10
RESULTS=/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/model_results
BENCH_CSV="$BASE/../data_analysis/all_radicals/project_5 - CCSDT_benchmarking.csv"
SCRATCH=/tmp/raptor_retrain_$$

mkdir -p "$OUTDIR" "$SCRATCH"
export OUTDIR SCRATCH
echo "=== RaPTOR retrain (80/10/10 split, trial_023 config, no BN) ==="
echo "Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started: $(date)"
echo ""

# ── 1. Train ─────────────────────────────────────────────────────────────────
echo "--- Training ---"
chemprop train \
  -i "$DATA_DIR/all_radicals.csv" \
  -s smiles \
  --atom-target-columns atom_y \
  --atom-features-path 0 "$DATA_DIR/all_radicals_atom_features.npz" \
  --splits-column split \
  -o "$OUTDIR" \
  --depth 11 \
  --message-hidden-dim 1500 \
  --ffn-hidden-dim 750 \
  --ffn-num-layers 2 \
  --atom-ffn-hidden-dim 750 \
  --atom-ffn-num-layers 1 \
  --dropout 0.1 \
  --ensemble-size 1 \
  --epochs 50 \
  --patience 20 \
  --max-lr 3.72e-4 \
  --batch-size 256 \
  --metrics mae rmse \
  --accelerator gpu --devices 1

TRAIN_EXIT=$?
echo "Training done (exit $TRAIN_EXIT): $(date)"
[ $TRAIN_EXIT -ne 0 ] && { echo "TRAIN FAILED"; exit $TRAIN_EXIT; }

MODEL=$OUTDIR/model_0/best.pt

# ── 2. Full dataset prediction ───────────────────────────────────────────────
echo ""
echo "--- Full dataset prediction ---"
chemprop predict \
  -i "$DATA_DIR/all_radicals.csv" \
  -s smiles \
  --atom-features-path 0 "$DATA_DIR/all_radicals_atom_features.npz" \
  --model-path "$MODEL" \
  -o "$SCRATCH/all_preds.csv"

[ $? -ne 0 ] && { echo "PREDICT (full) FAILED"; exit 1; }

python3 - <<PYEOF
import ast, math, os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("$DATA_DIR")
RESULTS  = Path("$RESULTS")
SCRATCH  = Path("$SCRATCH")

data = pd.read_csv(DATA_DIR / "all_radicals.csv")
pred = pd.read_csv(SCRATCH / "all_preds.csv")
pred_col = "atom_y.1" if "atom_y.1" in pred.columns else pred.columns[-1]

# Save full preds for figure generation
pred.to_csv("/tmp/raptor_80_10_10_all_preds.csv", index=False)
print("Saved /tmp/raptor_80_10_10_all_preds.csv")

rows = []
for split_name in ["train", "val", "test", None]:
    idx_list = (data.index[data["split"] == split_name].tolist()
                if split_name else data.index.tolist())
    label = split_name or "all"
    y_true, y_pred = [], []
    for i in idx_list:
        try:
            tl = ast.literal_eval(str(data["atom_y"].iloc[i]))
            pl = ast.literal_eval(str(pred[pred_col].iloc[i]))
        except Exception:
            continue
        rc = next((j for j, v in enumerate(tl) if not math.isnan(float(v))), None)
        if rc is None or rc >= len(pl): continue
        try:
            t, p = math.expm1(float(tl[rc])), math.expm1(float(pl[rc]))
        except Exception:
            continue
        if math.isnan(t) or math.isnan(p): continue
        y_true.append(t); y_pred.append(p)
    yt, yp = np.array(y_true), np.array(y_pred)
    err = yp - yt
    rows.append({"split": label, "n": len(yt),
                 "MAE_eV": round(float(np.mean(np.abs(err))), 4),
                 "RMSE_eV": round(float(np.sqrt(np.mean(err**2))), 4),
                 "bias_eV": round(float(np.mean(err)), 4)})

print(f"\n{'Split':8s} {'N':>8s} {'MAE':>8s} {'RMSE':>8s} {'Bias':>8s}")
print("-" * 45)
for r in rows:
    print(f"{r['split']:8s} {r['n']:>8d} {r['MAE_eV']:>8.4f} {r['RMSE_eV']:>8.4f} {r['bias_eV']:>+8.4f}")

RESULTS.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(RESULTS / "raptor_80_10_10_dataset_eval.csv", index=False)
print(f"\nSaved raptor_80_10_10_dataset_eval.csv")
PYEOF

# ── 3. CCSD(T) benchmark ─────────────────────────────────────────────────────
echo ""
echo "--- CCSD(T) benchmark ---"
python3 - <<'PYEOF'
import ast, subprocess, math, tempfile, os
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MODEL     = Path(os.environ["OUTDIR"]) / "model_0/best.pt"
BENCH_CSV = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/all_radicals/project_5 - CCSDT_benchmarking.csv")
RESULTS   = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/model_results")
SCRATCH   = Path(tempfile.mkdtemp(prefix="bench_raptor_"))

bench = pd.read_csv(BENCH_CSV)
rows, feat_arrays, rc_idxs, refs, smiles_out = [], [], [], [], []

for _, row in bench.iterrows():
    mol = Chem.MolFromSmiles(row["closed shell smiles"])
    if mol is None: continue
    mol_h = Chem.AddHs(mol)
    h_idx = int(row["hydrogen no"]) - 1
    if h_idx >= mol_h.GetNumAtoms(): continue
    if mol_h.GetAtomWithIdx(h_idx).GetSymbol() != "H": continue
    heavy_idx = list(mol_h.GetAtomWithIdx(h_idx).GetNeighbors())[0].GetIdx()
    rw = RWMol(mol_h)
    rw.RemoveAtom(h_idx)
    new_hi = heavy_idx if heavy_idx < h_idx else heavy_idx - 1
    rw.GetAtomWithIdx(new_hi).SetNumRadicalElectrons(1)
    try: Chem.SanitizeMol(rw)
    except: continue
    smi = Chem.MolToSmiles(Chem.RemoveHs(rw))
    mol_c = Chem.MolFromSmiles(smi)
    if mol_c is None: continue
    rc_idx = next((a.GetIdx() for a in mol_c.GetAtoms() if a.GetNumRadicalElectrons() > 0), None)
    if rc_idx is None: continue
    n = mol_c.GetNumAtoms()
    vals = ["nan"] * n; vals[rc_idx] = "0.0"
    feat = np.zeros((n, 1), dtype=np.float32); feat[rc_idx, 0] = 1.0
    rows.append({"smiles": smi, "atom_y": str(vals)})
    feat_arrays.append(feat)
    rc_idxs.append(rc_idx); refs.append(float(row["w"])); smiles_out.append(smi)

tmp_csv = SCRATCH / "input.csv"
tmp_npz = SCRATCH / "features.npz"
out_csv = SCRATCH / "preds.csv"
pd.DataFrame(rows).to_csv(tmp_csv, index=False)
np.savez(tmp_npz, *feat_arrays)
print(f"Prepared {len(rows)} / {len(bench)} molecules")

res = subprocess.run([
    "chemprop", "predict",
    "-i", str(tmp_csv), "-s", "smiles",
    "--atom-features-path", "0", str(tmp_npz),
    "--model-path", str(MODEL),
    "-o", str(out_csv),
], capture_output=True, text=True)

if res.returncode != 0:
    print("PREDICT FAILED:", res.stderr[-400:])
    raise SystemExit(1)

pred = pd.read_csv(out_csv)
pred_col = "atom_y.1" if "atom_y.1" in pred.columns else pred.columns[-1]

preds = []
for i, rc_idx in enumerate(rc_idxs):
    vals = ast.literal_eval(pred[pred_col].iloc[i])
    preds.append(math.expm1(float(vals[rc_idx])))

preds    = np.array(preds)
refs_arr = np.array(refs)
err      = preds - refs_arr
abs_err  = np.abs(err)
pct      = abs_err / np.abs(refs_arr) * 100

print(f"\n── RaPTOR (80/10/10, no BN) vs CCSD(T) (n={len(refs_arr)}) ──────────────")
print(f"  MAE         = {abs_err.mean():.4f} eV")
print(f"  RMSE        = {np.sqrt(np.mean(err**2)):.4f} eV")
print(f"  avg % diff  = {pct.mean():.2f}%")
print(f"  median %    = {np.median(pct):.2f}%")
print(f"  signed bias = {err.mean():+.4f} eV")

print(f"\nComparison (CCSD(T) benchmark):")
print(f"  stage3_best (90/9/1, no BN): MAE=0.0922 eV  avg%=5.44%  med%=3.94%")
print(f"  final_bn    (90/9/1, BN):    MAE=0.1024 eV  avg%=6.12%")
print(f"  raptor      (80/10/10, no BN): MAE={abs_err.mean():.4f} eV  avg%={pct.mean():.2f}%  med%={np.median(pct):.2f}%")

print(f"\nWorst 15:")
print(f"{'SMILES':45s} {'CCSD(T)':>8s} {'Pred':>8s} {'AbsErr':>8s} {'%':>6s}")
print("-" * 78)
for i in np.argsort(abs_err)[::-1][:15]:
    print(f"{smiles_out[i]:45s} {refs_arr[i]:8.4f} {preds[i]:8.4f} {abs_err[i]:8.4f} {pct[i]:6.1f}%")

RESULTS.mkdir(parents=True, exist_ok=True)
per_mol = pd.DataFrame({
    "smiles": smiles_out, "ccsd_t": refs_arr.tolist(),
    "pred_raptor": preds.tolist(),
    "abs_err": abs_err.tolist(), "pct_diff": pct.tolist(),
    "signed_err": err.tolist(),
})
per_mol.to_csv(RESULTS / "benchmark_ccsd_t_raptor_80_10_10.csv", index=False)
print(f"\nSaved benchmark_ccsd_t_raptor_80_10_10.csv")
PYEOF

echo ""
echo "=== Done: $(date) ==="
