from __future__ import annotations

"""
Builds chemprop-ready inputs for the full (C/N/O/S) radical philicity
experiment.

Split strategy: stratified random 80/10/10 by radical_type, using the
updated Butina clustering file. Butina clusters are mostly singletons
(median size=1), so a grouped split offers negligible leakage protection;
a stratified random split is more standard and ensures balanced C/N/O/S
proportions across train/val/test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

UNDERSAMPLE_CARBON = "--undersample-carbon" in sys.argv


def get_radical_center_symbol(radical_smiles: str):
    try:
        mol = Chem.MolFromSmiles(radical_smiles, sanitize=True)
        if mol is None:
            return None
        rad_atoms = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
        if len(rad_atoms) != 1:
            return None
        return rad_atoms[0].GetSymbol()
    except Exception:
        return None


def cluster_stratified_split(
    df: pd.DataFrame,
    cluster_col: str,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 0,
) -> pd.Series:
    """
    Within each cluster: shuffle molecules, then assign each one independently
    with p=[frac_train, frac_val, frac_test]. Deterministic proportional rounding
    causes clusters of size 2-9 to assign 0 molecules to val/test, skewing the
    overall split toward train. Per-molecule sampling avoids this bias and
    produces ~80/10/10 across the full dataset via the law of large numbers.
    """
    rng = np.random.default_rng(seed)
    frac_test = 1.0 - frac_train - frac_val
    result = pd.Series(index=df.index, dtype=object)

    for _, group in df.groupby(cluster_col):
        idx = group.index.to_numpy()
        n = len(idx)
        shuffled = idx[rng.permutation(n)]
        assignments = rng.choice(
            ["train", "val", "test"], size=n, p=[frac_train, frac_val, frac_test]
        )
        for mol_idx, label in zip(shuffled, assignments):
            result[mol_idx] = label

    return result


DATA_PATH = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/all_radicals/all_philicities_compiled_with_butina_clusters.csv")
_out_suffix = "_undersample" if UNDERSAMPLE_CARBON else ""
OUT_DIR = Path(f"/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_data_all{_out_suffix}")
OUT_CSV = OUT_DIR / "all_radicals.csv"
OUT_NPZ = OUT_DIR / "all_radicals_atom_features.npz"

SMOKE_TEST = "--smoke-test" in sys.argv
SEED = 0
VALID_ELEMENTS = {"carbon": "C", "nitrogen": "N", "oxygen": "O", "sulfur": "S"}


def cluster_deduplicate_carbon(
    df: pd.DataFrame,
    carbon_ratio: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Cap carbon training rows at carbon_ratio × heteroatom count, cluster-aware."""
    rng = np.random.default_rng(seed)
    symbols = df["radical_smiles"].apply(get_radical_center_symbol)
    is_carbon = (symbols == "C").values
    het_df = df[~is_carbon]
    carbon_df = df[is_carbon].copy()
    n_het = len(het_df)
    n_carbon = len(carbon_df)
    target = int(carbon_ratio * n_het)

    if n_carbon <= target:
        print(f"Carbon ({n_carbon}) already <= target ({target}), no undersampling needed.")
        return df.reset_index(drop=True)

    cluster_sizes = carbon_df.groupby("cluster_assignment").size()
    lo, hi = 1, int(cluster_sizes.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(cluster_sizes.clip(upper=mid).sum()) <= target:
            lo = mid
        else:
            hi = mid - 1
    cap = lo

    total_at_cap = int(cluster_sizes.clip(upper=cap).sum())
    deficit = target - total_at_cap
    over_cap = cluster_sizes[cluster_sizes > cap].index.tolist()
    bonus = set(rng.choice(over_cap, size=min(deficit, len(over_cap)), replace=False).tolist())

    kept_idx = []
    for cluster_id, group in carbon_df.groupby("cluster_assignment"):
        n = len(group)
        this_cap = cap + (1 if cluster_id in bonus else 0)
        if n <= this_cap:
            kept_idx.extend(group.index.tolist())
        else:
            sorted_group = group.sort_values("philicity")
            pick = np.round(np.linspace(0, n - 1, this_cap)).astype(int)
            kept_idx.extend(sorted_group.iloc[pick].index.tolist())

    result = pd.concat([het_df, carbon_df.loc[kept_idx]]).sort_index().reset_index(drop=True)
    print(f"Carbon deduplication (cluster-based, philicity-stratified, ratio={carbon_ratio}):")
    print(f"  C:  {n_carbon:>6} -> {len(kept_idx):>6}  (target {target}, cap/cluster={cap}+-1)")
    for sym in ["N", "O", "S"]:
        n = int((symbols[~is_carbon] == sym).sum())
        print(f"  {sym}:  {n:>6} -> {n:>6}  (unchanged)")
    return result


def canonicalize_no_h(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def radical_center_atom_index(mol: Chem.Mol) -> int | None:
    rad_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    return rad_idx[0] if len(rad_idx) == 1 else None


def build_atom_y_str(n_atoms: int, rc_idx: int, target_value: float) -> str:
    vals: list = ["nan"] * n_atoms
    vals[rc_idx] = float(target_value)
    return str(vals)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    n_total = len(df)
    print(f"Loaded {n_total} rows from {DATA_PATH}")

    rdkit_sym = df["radical_smiles"].apply(get_radical_center_symbol)
    col_sym = df["radical_type"].str.lower().map(VALID_ELEMENTS)
    mismatch = (rdkit_sym != col_sym) & col_sym.notna() & rdkit_sym.notna()
    print(f"radical_type vs RDKit-derived symbol: {mismatch.sum()} mismatches out of "
          f"{(col_sym.notna() & rdkit_sym.notna()).sum()} comparable rows")

    df = df[(df["radical_type"].str.lower().isin(VALID_ELEMENTS)) & (rdkit_sym == col_sym)].reset_index(drop=True)
    print(f"Filtered to C/N/O/S radicals (both methods agree): {n_total} -> {len(df)} rows")
    print(df["radical_type"].str.lower().value_counts())

    if SMOKE_TEST:
        df = df.sample(n=min(200, len(df)), random_state=SEED).reset_index(drop=True)
        print(f"--smoke-test: subsampled to {len(df)} rows")

    # Cluster-stratified 80/10/10: within each Butina cluster, assign
    # molecules proportionally to train/val/test. Singletons are randomly
    # assigned with 80/10/10 probability.
    df["split"] = cluster_stratified_split(
        df, cluster_col="butina_cluster_assignment", seed=SEED
    )
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    if UNDERSAMPLE_CARBON:
        print("\nUndersampling carbon radicals in training set (carbon_ratio=2.0)...")
        train_df = cluster_deduplicate_carbon(train_df, carbon_ratio=2.0, seed=SEED)

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
    print(pd.crosstab(df["radical_type"], df["split"], normalize="columns").round(3))

    rows = []
    atom_feat_arrays = []
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

    print(f"Dropped {n_dropped} rows (invalid SMILES or ambiguous radical center after canonicalization)")

    out_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    np.savez(OUT_NPZ, *atom_feat_arrays)
    print(f"Wrote {len(out_df)} rows to {OUT_CSV}")
    print(f"Wrote {len(atom_feat_arrays)} per-molecule atom-feature arrays to {OUT_NPZ}")
    print(out_df["split"].value_counts())
    print(pd.crosstab(out_df["radical_type"], out_df["split"]))


if __name__ == "__main__":
    main()
