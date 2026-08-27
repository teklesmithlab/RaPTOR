from __future__ import annotations

"""
Export true zero-shot predictions for plotting.

For each heteroatom E, take the `drop_E` model -- which never saw a single
radical centred on E, in its representation or its readout -- and pull its
predictions on E's own test rows. This is the honest version of what the
frozen-embedding probe was estimating.

Cationic N is excluded. 105 cationic molecules span 5.15-11.98 eV while
everything else the model ever saw lives at 0.5-4 eV, so it places them ~7.8 eV
too low; that 6% of rows flips the pooled nitrogen correlation from +0.82 to
-0.07. Charge state is a separate transfer problem and is reported separately.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import loeo_evaluate as L

OUT = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/"
           "data_analysis/model_results/loeo_zeroshot_predictions.csv")
ELEMENTS = ["nitrogen", "oxygen", "sulfur"]


def is_cation(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    r = [a for a in m.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    return len(r) == 1 and r[0].GetFormalCharge() > 0


def main():
    rows, summary = [], []
    for e in ELEMENTS:
        fold = f"drop_{e}"
        dat = pd.read_csv(L.DATA_ROOT / fold / "all_radicals.csv")
        pred = pd.read_csv(L.RUN_ROOT / fold / "model_0" / "test_predictions.csv")
        test = dat[dat.split == "test"].reset_index(drop=True)

        idx = [L.radical_idx(s) for s in test.smiles]
        yt = np.expm1(np.array([L.center_value(c, i) for c, i in zip(test.atom_y, idx)]))
        yp = np.expm1(np.array([L.center_value(c, i) for c, i in zip(pred.atom_y, idx)]))

        keep = ((test.radical_type == e).to_numpy()
                & np.isfinite(yt) & np.isfinite(yp)
                & ~np.array([is_cation(s) for s in test.smiles]))
        a, b = yt[keep], yp[keep]

        r = float(np.corrcoef(a, b)[0, 1])
        r2 = float(1 - np.sum((b - a) ** 2) / np.sum((a - a.mean()) ** 2))
        bias = float(np.mean(b - a))
        slope = float(np.polyfit(b, a, 1)[0])
        print(f"{e:<9} n={keep.sum():>5}  r={r:>6.3f}  R2={r2:>7.3f}  "
              f"bias={bias:>+6.3f} eV  MAE={np.mean(np.abs(b-a)):.3f}")

        rows.append(pd.DataFrame(dict(element=e, y_true=a, y_pred=b)))
        summary.append(dict(element=e, n=int(keep.sum()), pearson_r=r,
                            r2_raw=r2, bias_ev=bias, inv_slope=slope,
                            mae=float(np.mean(np.abs(b - a)))))

    pd.concat(rows).to_csv(OUT, index=False)
    pd.DataFrame(summary).to_csv(OUT.with_name("loeo_zeroshot_summary.csv"), index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
