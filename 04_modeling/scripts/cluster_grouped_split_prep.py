from __future__ import annotations

"""
Builds a single 80/10/10 train/val/test split where clusters (not molecules)
are held out entirely -- i.e. no cluster ever appears in more than one of
{train, val, test}. This is the direct counterpart to the primary model's
stratified 80/10/10 split (chemprop_all_prep.py, which spreads each cluster's
molecules proportionally across train/val/test): same architecture, same
clustering source (butina_cluster_assignment), same target fractions --
differing in exactly one variable, whether clusters are grouped or stratified,
to isolate that as the generalization-gap comparison.

Split mechanics: reuses the same balanced bin-packing as loco_cv_prep.py --
clusters assigned to 10 equal-target bins by greedy LPT (largest cluster
first, each to the currently-smallest bin), done per radical_type. Bin 0 is
held out as test, bin 1 as val, bins 2-9 combined as train, giving ~80/10/10
by construction without needing separate bin-size logic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chemprop_all_prep import (
    get_radical_center_symbol,
    canonicalize_no_h,
    radical_center_atom_index,
    build_atom_y_str,
    VALID_ELEMENTS,
)
from loco_cv_prep import assign_folds_balanced

DATA_PATH = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/all_radicals/all_philicities_compiled_with_butina_clusters.csv")
OUT_DIR = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_data_cluster_grouped")
CLUSTER_COL = "butina_cluster_assignment"

N_BINS = 10  # bin 0 -> test, bin 1 -> val, bins 2-9 -> train  (~80/10/10)
SEED = 0


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    rdkit_sym = df["radical_smiles"].apply(get_radical_center_symbol)
    col_sym = df["radical_type"].str.lower().map(VALID_ELEMENTS)
    df = df[(df["radical_type"].str.lower().isin(VALID_ELEMENTS)) & (rdkit_sym == col_sym)].reset_index(drop=True)
    print(f"Filtered to C/N/O/S radicals (both methods agree): {len(df)} rows")

    df["bin"] = assign_folds_balanced(df, CLUSTER_COL, N_BINS, SEED)
    df["split"] = np.where(df["bin"] == 0, "test",
                   np.where(df["bin"] == 1, "val", "train"))
    print(f"Split sizes: {df['split'].value_counts().to_dict()}")
    print(f"Fraction: {(df['split'].value_counts(normalize=True)*100).round(2).to_dict()}")
    print(pd.crosstab(df["radical_type"], df["split"], normalize="columns").round(3))

    rows, atom_feat_arrays = [], []
    n_dropped = 0
    for _, row in df.iterrows():
        smi = canonicalize_no_h(row["radical_smiles"])
        if smi is None:
            n_dropped += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        rc_idx = radical_center_atom_index(mol)
        if rc_idx is None:
            n_dropped += 1
            continue

        n_atoms = mol.GetNumAtoms()
        log_philicity = float(np.log1p(row["philicity"]))
        atom_y = build_atom_y_str(n_atoms, rc_idx, log_philicity)

        flag = np.zeros((n_atoms, 1), dtype=np.float32)
        flag[rc_idx, 0] = 1.0
        atom_feat_arrays.append(flag)

        rows.append({"smiles": smi, "atom_y": atom_y, "split": row["split"], "radical_type": row["radical_type"]})

    print(f"Dropped {n_dropped} rows (invalid SMILES / ambiguous radical center)")

    out_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_DIR / "all_radicals.csv", index=False)
    np.savez(OUT_DIR / "all_radicals_atom_features.npz", *atom_feat_arrays)
    print(f"Wrote {len(out_df)} rows to {OUT_DIR / 'all_radicals.csv'}")
    print(out_df["split"].value_counts())


if __name__ == "__main__":
    main()
