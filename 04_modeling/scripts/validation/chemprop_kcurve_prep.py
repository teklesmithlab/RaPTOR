from __future__ import annotations

"""
Build the folds for a genuine end-to-end learning curve on a NEW element.

Panel F previously used a linear probe on a frozen embedding, which measures
how cheaply a readout adapts -- not how much data a new element actually
needs, because the embedding had already seen that element. This replaces it
with retraining.

For each heteroatom E and each k:
    train = every training row whose element is NOT E,  plus k rows of E
    val   = validation rows whose element is NOT E      (so early stopping
            never touches E; matches the drop_E folds)
    test  = untouched, all four elements

k = 0 is exactly the existing `drop_E` fold and k = "all" is generated here
rather than reusing `baseline`, because baseline's validation set contains E
and would break the convention above.

Small k is noisy -- which k molecules you happen to draw matters a lot -- so
k <= 100 gets two seeds.

Source of rows is chemprop_data_loeo/baseline, which already carries the
stereo-grouped split and log1p targets.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling")
SRC = BASE / "chemprop_data_loeo/baseline"
OUT_ROOT = BASE / "chemprop_data_kcurve"

ELEMENTS = ["nitrogen", "oxygen", "sulfur"]
KS = [5, 25, 100, 500, 2000, "all"]
SEEDS = {5: [0, 1], 25: [0, 1], 100: [0, 1]}      # others get one seed
DEFAULT_SEEDS = [0]


def main():
    df = pd.read_csv(SRC / "all_radicals.csv")
    feats = np.load(SRC / "all_radicals_atom_features.npz")
    arrs = [feats[k] for k in feats.files]
    assert len(arrs) == len(df), f"{len(arrs)} feature arrays vs {len(df)} rows"
    print(f"source: {len(df)} rows  "
          f"{df.split.value_counts().to_dict()}")

    configs = []
    for e in ELEMENTS:
        pool = np.where((df.split == "train").to_numpy()
                        & (df.radical_type == e).to_numpy())[0]
        for k in KS:
            for seed in (SEEDS.get(k, DEFAULT_SEEDS)):
                rng = np.random.default_rng(1000 * seed + hash(e) % 997)
                if k == "all":
                    chosen = pool
                    name = f"{e}_kall_s{seed}"
                else:
                    if k > len(pool):
                        continue
                    chosen = rng.choice(pool, size=k, replace=False)
                    name = f"{e}_k{k}_s{seed}"

                keep = np.zeros(len(df), dtype=bool)
                # everything that is not this element, in train and val
                keep |= ((df.split != "test").to_numpy()
                         & (df.radical_type != e).to_numpy())
                # the k sampled molecules of this element (train only)
                keep[chosen] = True
                # full test set
                keep |= (df.split == "test").to_numpy()

                sub = df[keep].reset_index(drop=True)
                sub_arrs = [a for a, m in zip(arrs, keep) if m]

                d = OUT_ROOT / name
                d.mkdir(parents=True, exist_ok=True)
                sub.to_csv(d / "all_radicals.csv", index=False)
                np.savez(d / "all_radicals_atom_features.npz", *sub_arrs)

                n_e_train = int(((sub.split == "train")
                                 & (sub.radical_type == e)).sum())
                configs.append(dict(name=name, element=e,
                                    k=(len(pool) if k == "all" else k), seed=seed,
                                    n_train=int((sub.split == "train").sum()),
                                    n_element_train=n_e_train))
                print(f"  {name:<22} train={configs[-1]['n_train']:>7}  "
                      f"{e[:1].upper()}_in_train={n_e_train:>6}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(configs).to_csv(OUT_ROOT / "config_summary.csv", index=False)
    (OUT_ROOT / "fold_names.json").write_text(
        json.dumps([c["name"] for c in configs], indent=1))
    print(f"\n{len(configs)} folds written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
