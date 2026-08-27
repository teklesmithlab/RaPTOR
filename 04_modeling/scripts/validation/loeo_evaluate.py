from __future__ import annotations

"""
Evaluate the leave-one-element-out folds.

This is the experiment that removes the standing caveat on the probe analysis.
In `drop_nitrogen`, the network never saw a single nitrogen radical -- not in
its representation, not in its readout -- so its performance on nitrogen is a
genuine zero-shot number rather than a statement about how cheaply a readout
adapts on a jointly-trained embedding.

Two comparisons matter:

1. baseline vs drop_X, measured on element X. How much does an element lose
   when it is removed entirely? This is the zero-shot transfer result.

2. drop_carbon vs sizematch_carbon. Both train on 39,172 molecules; the first
   has no carbon at all, the second is the full four-element pool downsampled
   to the same count. Any gap is attributable to what carbon *is* rather than
   how much of it there is -- which is the control the "carbon is 61% of the
   data, of course removing it hurts" objection demands.

Targets are log1p(philicity); everything is converted back to eV before
reporting so the numbers are comparable to the production MAE.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

BASE = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/modeling")
DATA_ROOT = BASE / "chemprop_data_loeo"
RUN_ROOT = BASE / "chemprop_runs/loeo"
OUT = BASE.parent / "data_analysis/model_results/loeo_results.csv"

FOLDS = ["baseline", "sizematch_carbon", "drop_carbon",
         "drop_nitrogen", "drop_oxygen", "drop_sulfur"]
ELEMENTS = ["carbon", "nitrogen", "oxygen", "sulfur"]
SYM = {"carbon": "C", "nitrogen": "N", "oxygen": "O", "sulfur": "S"}


def radical_idx(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    r = [a.GetIdx() for a in m.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    return r[0] if len(r) == 1 else None


def center_value(cell, idx):
    """Pull the radical-centre entry out of a per-atom list."""
    if idx is None:
        return np.nan
    try:
        vals = ast.literal_eval(str(cell))
    except Exception:
        try:
            vals = [float(v) for v in str(cell).strip("[]").split(",")]
        except Exception:
            return np.nan
    if idx >= len(vals):
        return np.nan
    v = vals[idx]
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_fold(fold):
    dat = pd.read_csv(DATA_ROOT / fold / "all_radicals.csv")
    pred = pd.read_csv(RUN_ROOT / fold / "model_0" / "test_predictions.csv")
    test = dat[dat.split == "test"].reset_index(drop=True)
    if len(test) != len(pred):
        raise SystemExit(f"{fold}: {len(test)} test rows vs {len(pred)} predictions")

    idx = [radical_idx(s) for s in test.smiles]
    y_true = np.array([center_value(c, i) for c, i in zip(test.atom_y, idx)])
    y_pred = np.array([center_value(c, i) for c, i in zip(pred.atom_y, idx)])
    # targets are log1p(philicity)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return pd.DataFrame({
        "radical_type": test.radical_type[ok].to_numpy(),
        "true_ev": np.expm1(y_true[ok]),
        "pred_ev": np.expm1(y_pred[ok]),
    })


def metrics(d):
    e = d.pred_ev - d.true_ev
    ss = np.sum((d.true_ev - d.true_ev.mean()) ** 2)
    return dict(n=len(d), mae=float(np.mean(np.abs(e))),
                rmse=float(np.sqrt(np.mean(e ** 2))),
                r2=float(1 - np.sum(e ** 2) / ss) if ss > 0 else np.nan,
                r=float(np.corrcoef(d.true_ev, d.pred_ev)[0, 1]))


def main():
    rows = []
    per_fold = {}
    for f in FOLDS:
        d = load_fold(f)
        per_fold[f] = d
        for e in ELEMENTS:
            sub = d[d.radical_type == e]
            if len(sub) < 20:
                continue
            rows.append(dict(fold=f, element=e, **metrics(sub)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    piv_mae = res.pivot(index="fold", columns="element", values="mae")
    piv_r2 = res.pivot(index="fold", columns="element", values="r2")

    print("=" * 78)
    print("TEST MAE (eV), by fold x element   [identical test set across folds]")
    print("=" * 78)
    print(piv_mae.reindex(FOLDS).round(4).to_string())
    print("\n" + "=" * 78)
    print("TEST R^2, by fold x element")
    print("=" * 78)
    print(piv_r2.reindex(FOLDS).round(4).to_string())

    print("\n" + "=" * 78)
    print("1. ZERO-SHOT: element removed entirely from training")
    print("=" * 78)
    print(f"{'element':<10}{'baseline MAE':>14}{'zero-shot MAE':>15}{'ratio':>8}"
          f"{'baseline R2':>13}{'zero-shot R2':>14}")
    for e in ["carbon", "nitrogen", "oxygen", "sulfur"]:
        f = f"drop_{e}"
        if f not in piv_mae.index or e not in piv_mae.columns:
            continue
        b, z = piv_mae.loc["baseline", e], piv_mae.loc[f, e]
        br, zr = piv_r2.loc["baseline", e], piv_r2.loc[f, e]
        print(f"{SYM[e]:<10}{b:>14.4f}{z:>15.4f}{z/b:>8.2f}x{br:>12.3f}{zr:>14.3f}")

    print("\n" + "=" * 78)
    print("2. SIZE-MATCHED CARBON CONTROL (both train on 39,172 molecules)")
    print("=" * 78)
    print(f"{'element':<10}{'no carbon':>12}{'size-matched':>14}{'difference':>12}")
    for e in ELEMENTS:
        if e not in piv_mae.columns:
            continue
        dc, sm = piv_mae.loc["drop_carbon", e], piv_mae.loc["sizematch_carbon", e]
        print(f"{SYM[e]:<10}{dc:>12.4f}{sm:>14.4f}{dc - sm:>+12.4f}")
    print("\nA positive difference means the no-carbon model is worse at equal training size,")
    print("i.e. carbon contributes something beyond sheer volume.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
