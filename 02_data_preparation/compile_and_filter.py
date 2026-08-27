"""
Compile all batch_NNN_philicities.csv files, apply the current filtering
pipeline from preprocess_for_modeling.py, and report radical type breakdown.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, '/insomnia001/depts/tekle_smith/users/MKL/project_5/data_preparation')
from preprocess_for_modeling import (
    filter_philicity,
    filter_unassigned_stereo_rows,
    filter_other_radicals,
    filter_individual_exclusions,
    filter_ring_sulfurane_artifacts,
    filter_spin_contamination,
    filter_unphysical_energies,
    filter_acyloxy,
    filter_phosphonate_artifacts,
    filter_aryl_artifacts,
    filter_halothiophene_artifacts,
    filter_si_cation_artifacts,
    filter_halogen_cation_artifacts,
    filter_n_radical_peroxide_artifacts,
)

BATCHES_DIR = Path('/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/batches')
OUT_CSV = Path('/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/all_radicals/all_philicities_compiled.csv')


# ── 1. Compile ────────────────────────────────────────────────────────────────

print("Reading batch philicities files...")
frames = []
for path in sorted(BATCHES_DIR.glob("batch_*_philicities.csv")):
    if path.suffix != '.csv':
        continue
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  Could not read {path.name}: {e}")
        continue

    # extract batch number from filename
    stem = path.stem  # e.g. "batch_042_philicities"
    batch_num = stem.split('_')[1]

    # normalise column names across old and new formats
    # old: timestamp, radical_name, cluster_assignment, radical_smiles, philicity, I, A
    # new: radical_number, timestamp, original_smiles, radical_name,
    #      cluster_assignment, radical_smiles, philicity, I, A, S2

    # add batch_id if not present
    if 'batch_id' not in df.columns:
        df['batch_id'] = batch_num

    # add missing columns as NaN so concat works cleanly
    for col in ['original_smiles', 'radical_number', 'S2', 'timestamp']:
        if col not in df.columns:
            df[col] = pd.NA

    frames.append(df)

if not frames:
    print("No batch files found — exiting")
    sys.exit(1)

raw = pd.concat(frames, ignore_index=True)
print(f"  Compiled {len(raw):,} rows from {len(frames)} batch files\n")

# drop rows missing the core columns
required = ['radical_smiles', 'philicity', 'I', 'A']
raw = raw.dropna(subset=required).reset_index(drop=True)
print(f"  After dropping rows with missing core values: {len(raw):,}\n")

# deduplicate on (radical_name, radical_smiles) keeping first occurrence
n_before = len(raw)
raw = raw.drop_duplicates(subset=['radical_name', 'radical_smiles']).reset_index(drop=True)
print(f"  After deduplication: {len(raw):,} (removed {n_before - len(raw):,} duplicates)\n")


# ── 2. Apply filters ──────────────────────────────────────────────────────────

print("=" * 60)
print("Applying filters")
print("=" * 60)

df = raw.copy()

# unassigned stereochemistry — fall back to radical_smiles if original_smiles absent
if df['original_smiles'].notna().any():
    df = filter_unassigned_stereo_rows(df, smiles_col='original_smiles')
else:
    print("No original_smiles column — skipping stereo filter")

# radical centre must be C, N, O, or S
df = filter_other_radicals(df)

# hand-identified individual exclusions + invalid ring-sulfurane structures
df = filter_individual_exclusions(df)
df = filter_ring_sulfurane_artifacts(df)

# spin contamination (uses S2 column when available, DFT files otherwise)
df = filter_spin_contamination(df)

# unphysical I / A values (fast, no external deps)
df = filter_unphysical_energies(df)

# DFT artifact filters (all work purely on radical_smiles + I/A columns)
df = filter_acyloxy(df)
df = filter_phosphonate_artifacts(df)
df = filter_aryl_artifacts(df)
df = filter_halothiophene_artifacts(df)
df = filter_si_cation_artifacts(df)
df = filter_halogen_cation_artifacts(df)
df = filter_n_radical_peroxide_artifacts(df)

# philicity in [0.2, 4.0] eV
df = filter_philicity(df)

# live-DFT-parsing filters skipped (filter_missing_dft, filter_radical_migrated,
# filter_phosphonate_o_radical_artifacts) — new radicals are not in the v21 localization
# CSV / don't yet have a matched DFT output directory, so those filters would wrongly
# drop valid new data

print(f"\nFinal dataset: {len(df):,} rows\n")


# ── 3. Radical type breakdown ─────────────────────────────────────────────────

def classify_radical_type(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "other"
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return {"C": "carbon", "N": "nitrogen", "O": "oxygen", "S": "sulfur"}.get(
                atom.GetSymbol(), "other"
            )
    return "other"

df['radical_type'] = df['radical_smiles'].apply(classify_radical_type)

ORDER = ["carbon", "nitrogen", "oxygen", "sulfur", "other"]
total = len(df)

print("=" * 60)
print("Radical type breakdown (post-filter)")
print("=" * 60)
print(f"{'type':<12} {'count':>8}  {'%':>6}")
print("-" * 32)
for rtype in ORDER:
    n = int((df['radical_type'] == rtype).sum())
    if n > 0:
        print(f"{rtype:<12} {n:>8,}  {100*n/total:>5.1f}%")
print("-" * 32)
print(f"{'TOTAL':<12} {total:>8,}  100.0%")

# also show philicity distribution per type
print("\nPhilicity statistics per type:")
print(f"{'type':<12} {'mean':>8}  {'median':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
print("-" * 60)
for rtype in ORDER:
    sub = df[df['radical_type'] == rtype]['philicity']
    if len(sub) == 0:
        continue
    print(f"{rtype:<12} {sub.mean():>8.3f}  {sub.median():>8.3f}  {sub.std():>8.3f}  {sub.min():>8.3f}  {sub.max():>8.3f}")


# ── 4. Save ───────────────────────────────────────────────────────────────────

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved compiled + filtered dataset to:\n  {OUT_CSV}")
