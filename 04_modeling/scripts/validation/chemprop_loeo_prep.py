from __future__ import annotations

"""
Leave-one-element-out folds, plus the size-matched control that makes the
carbon fold interpretable.

The obvious objection to a carbon ablation is that carbon is 61% of the data,
so removing it must hurt simply because the training set shrank. That
objection is correct, and it is why `sizematch_carbon` exists: it trains on
all four elements but downsamples to exactly the row count of the no-carbon
set. Comparing `drop_carbon` against `sizematch_carbon` holds dataset size
fixed, so any remaining gap is attributable to what carbon *is* rather than
how much of it there is.

The heteroatom folds do not suffer that confound at all -- N, O and S are each
only 9-16% of the data, so dropping one barely changes the training size while
removing the target element entirely. Those are genuine zero-shot tests.

Folds written:
  baseline          all four elements (reference)
  drop_carbon       train/val carry no carbon
  drop_nitrogen     train/val carry no nitrogen
  drop_oxygen       train/val carry no oxygen
  drop_sulfur       train/val carry no sulfur
  sizematch_carbon  all elements, train downsampled to |drop_carbon train|

In every fold the TEST split is left untouched and contains all four elements,
so a single evaluation pass measures both held-out-element performance and any
collateral damage to the elements that were kept.

Target is log1p(philicity), matching chemprop_stereo_prep.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from chemprop_stereo_prep import (  # noqa: E402
    DATA_PATH, CLUSTER_COL, SEED,
    canonicalize_no_h, stereo_stripped, radical_center_atom_index,
    build_atom_y_str, stereo_grouped_split,
)

OUT_ROOT = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling/chemprop_data_loeo")
ELEMENTS = ["carbon", "nitrogen", "oxygen", "sulfur"]
SMOKE = "--smoke-test" in sys.argv


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df.radical_type.isin(ELEMENTS)].copy()
    df["family"] = df["radical_smiles"].map(stereo_stripped)
    df = df.dropna(subset=["family", CLUSTER_COL]).reset_index(drop=True)

    if SMOKE:
        keep = df["family"].drop_duplicates().sample(n=400, random_state=SEED)
        df = df[df["family"].isin(keep)].reset_index(drop=True)

    df["split"] = stereo_grouped_split(df, cluster_col=CLUSTER_COL,
                                       family_col="family", seed=SEED)
    straddle = df.groupby("family")["split"].nunique()
    assert (straddle == 1).all(), f"{int((straddle > 1).sum())} families straddle the split"
    print(f"{len(df)} rows, {df.family.nunique()} families; split verified non-straddling")
    print(pd.crosstab(df["radical_type"], df["split"]).to_string(), "\n")

    # featurize once -- atom_y and the radical flag do not depend on the fold
    rows, feats, n_dropped = [], [], 0
    for row in df.itertuples():
        smi = canonicalize_no_h(row.radical_smiles)
        if smi is None:
            n_dropped += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        rc = radical_center_atom_index(mol)
        if rc is None:
            n_dropped += 1
            continue
        n_atoms = mol.GetNumAtoms()
        flag = np.zeros((n_atoms, 1), dtype=np.float32)
        flag[rc, 0] = 1.0
        feats.append(flag)
        rows.append({"smiles": smi,
                     "atom_y": build_atom_y_str(n_atoms, rc, float(np.log1p(row.philicity))),
                     "split": row.split, "radical_type": row.radical_type})
    base = pd.DataFrame(rows)
    print(f"featurized {len(base)} rows ({n_dropped} dropped)\n")

    n_no_carbon_train = int(((base.split == "train") & (base.radical_type != "carbon")).sum())
    rng = np.random.default_rng(SEED)

    folds = {"baseline": None, "sizematch_carbon": "sizematch"}
    folds.update({f"drop_{e}": e for e in ELEMENTS})

    summary = []
    for name, spec in folds.items():
        keep = np.ones(len(base), dtype=bool)
        if spec in ELEMENTS:
            # remove that element from train and val only; test keeps everything
            keep &= ~((base.radical_type == spec) & (base.split != "test")).to_numpy()
        elif spec == "sizematch":
            tr_idx = np.where((base.split == "train").to_numpy())[0]
            drop = rng.choice(tr_idx, size=len(tr_idx) - n_no_carbon_train, replace=False)
            keep[drop] = False

        sub = base[keep].reset_index(drop=True)
        sub_feats = [f for f, k in zip(feats, keep) if k]
        d = OUT_ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        sub.to_csv(d / "all_radicals.csv", index=False)
        np.savez(d / "all_radicals_atom_features.npz", *sub_feats)

        ntr = int((sub.split == "train").sum())
        print(f"{name:<18} rows={len(sub):>7}  train={ntr:>7}  "
              f"val={int((sub.split=='val').sum()):>6}  test={int((sub.split=='test').sum()):>6}")
        summary.append(dict(fold=name, rows=len(sub), train=ntr,
                            val=int((sub.split == "val").sum()),
                            test=int((sub.split == "test").sum())))

    pd.DataFrame(summary).to_csv(OUT_ROOT / "fold_summary.csv", index=False)
    print(f"\nno-carbon train size = {n_no_carbon_train} (sizematch_carbon matches this)")
    print(f"wrote folds to {OUT_ROOT}")


if __name__ == "__main__":
    main()
