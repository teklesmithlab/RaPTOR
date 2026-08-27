#!/usr/bin/env python3
"""Assemble raptor_dataset.csv — the published 142,779-radical dataset.

This is the script that produced the CSV sitting next to it. It joins three
sources that live in the original working tree (paths below), none of which is
shipped in this directory:

  1. the compiled DFT table   — original_smiles, radical_smiles, I, A, philicity
  2. the chemprop model input — fixes the published row order
  3. the paper-split model's predictions — the predicted_philicity column

Why a join and not a column drop
--------------------------------
The compiled DFT table and the chemprop inputs use *different* RDKit SMILES
canonicalisations for the same molecule -- e.g. `[CH2]c1c([nH]nc1)Br` in the
former against `[CH2]c1cn[nH]c1Br` in the latter. Matching on the raw SMILES
string recovers only about a third of the rows. Every join here is therefore
keyed on (canonical SMILES, philicity rounded to 9 dp).

Philicity is carried as the key rather than the SMILES alone because 7,224
SMILES appear in more than one row of the compiled table (see the top-level
README). Three philicity values collide between genuinely different molecules,
which is why the canonical SMILES is in the key as well -- neither field is
sufficient on its own.

Row order follows source 2, so `splits.json` indexes the published CSV directly.

The target stored in the model inputs is log1p(philicity); both the truth and
the prediction are passed through expm1 here to return eV.

Verified on the shipped output: all 142,779 rows joined with zero misses;
philicity reproduces from I and A via the Parr formula to within 7.1e-15 eV;
and the splits.json val/test indices select the 12,695 and 1,353 structures
that match val_preds.csv and test_predictions.csv in order.
"""
from __future__ import annotations

import ast
import collections
import csv
import math
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

P5 = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5")
HERE = Path(__file__).resolve().parent

COMPILED = P5 / "data_analysis/all_radicals/all_philicities_compiled_with_capped_clusters.csv"
MODEL_IN = P5 / "modeling/chemprop_data_all_undersample/all_radicals.csv"
PREDS    = P5 / "modeling/chemprop_runs/stratified_80_10_10/all_preds.csv"
OUT      = HERE / "raptor_dataset.csv"

_canon_cache: dict[str, str | None] = {}


def canon(smiles: str) -> str | None:
    """RDKit canonical SMILES, or None if the string will not parse."""
    if smiles in _canon_cache:
        return _canon_cache[smiles]
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        out = Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        out = None
    _canon_cache[smiles] = out
    return out


def center_value(cell: str) -> tuple[int, float]:
    """Index and value of the single non-nan entry in an atom_y list."""
    values = ast.literal_eval(cell)
    idx = [i for i, v in enumerate(values) if not (isinstance(v, str) and v == "nan")]
    if len(idx) != 1:
        raise ValueError(f"expected exactly one target atom, got {len(idx)}")
    return idx[0], float(values[idx[0]])


def main() -> int:
    # 1. compiled DFT table: source of original_smiles / I / A / philicity
    by_canon: dict[tuple[str, float], dict] = {}
    by_philicity: dict[float, list[dict]] = collections.defaultdict(list)
    with open(COMPILED) as fh:
        for row in csv.DictReader(fh):
            try:
                phil = float(row["philicity"])
            except (TypeError, ValueError):
                continue
            key = round(phil, 9)
            by_philicity[key].append(row)
            c = canon(row["radical_smiles"])
            if c is not None:
                by_canon[(c, key)] = row
    print(f"compiled: {sum(len(v) for v in by_philicity.values())} rows, {len(by_canon)} canonical keys")

    # 2. predictions from the paper-split model (covers all rows, reshuffled order)
    predictions: dict[tuple[str, float], float] = {}
    with open(PREDS) as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            idx, truth = center_value(row[1])
            preds = ast.literal_eval(row[4])
            c = canon(row[0])
            if c is not None:
                predictions[(c, round(math.expm1(truth), 9))] = math.expm1(float(preds[idx]))
    print(f"predictions: {len(predictions)} keys")

    # 3. emit in model-input row order, so splits.json still indexes the result
    written = unmatched_source = unmatched_pred = 0
    with open(MODEL_IN) as fin, open(OUT, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["original_smiles", "radical_smiles", "I", "A",
                         "philicity", "predicted_philicity"])
        for row in csv.DictReader(fin):
            _, truth = center_value(row["atom_y"])
            key = round(math.expm1(truth), 9)
            c = canon(row["smiles"])

            source = by_canon.get((c, key))
            if source is None:                      # fall back to a unique philicity match
                candidates = by_philicity.get(key, [])
                source = candidates[0] if len(candidates) == 1 else None
            if source is None:
                unmatched_source += 1
                continue

            predicted = predictions.get((c, key))
            if predicted is None:
                unmatched_pred += 1
                continue

            writer.writerow([source["original_smiles"], source["radical_smiles"],
                             source["I"], source["A"], source["philicity"], repr(predicted)])
            written += 1

    print(f"wrote {written} rows -> {OUT}")
    if unmatched_source or unmatched_pred:
        print(f"UNJOINED: compiled={unmatched_source} predictions={unmatched_pred}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
