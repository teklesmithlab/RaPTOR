# 03 — Dataset

`build_raptor_dataset.py` generates `raptor_dataset.csv` from the DFT table and the
model's predictions. It reads its three inputs by absolute path from the original working
tree; re-running it reproduces the shipped CSV byte-for-byte.

`raptor_dataset.csv` — **142,779 radicals**, the dataset the RaPTOR model was trained
and evaluated on. One row per radical.

| Column | Meaning |
|---|---|
| `original_smiles` | the closed-shell parent molecule |
| `radical_smiles` | the radical, with the radical centre marked |
| `I` | vertical ionization potential, eV |
| `A` | vertical electron affinity, eV |
| `philicity` | Parr electrophilicity index ω = (I + A)² / (8(I − A)), eV — the DFT reference value |
| `predicted_philicity` | the model's prediction, eV |

`I` and `A` come from single-point calculations on the cation and anion at the optimised
neutral geometry. The `philicity` column is exactly reproducible from `I` and `A` by the
formula above — verified for all 142,779 rows to within 7.1e-15 eV.

Composition: carbon 90,927 / sulfur 20,511 / nitrogen 18,495 / oxygen 12,846.

## Row order

Row order is significant. It matches `04_modeling/production_model/training_inputs/`, so
the index lists in `splits.json` select the same molecules from this file. Verified: the
`val` indices reproduce the 12,695 validation structures in order, and the `test` indices
the 1,353 test structures, matching `val_preds.csv` and `test_predictions.csv` exactly.

## About `predicted_philicity`

These predictions come from the model in `04_modeling/paper_split_model/`, which is the
one used for the manuscript's parity and error panels. It covers all 142,779 rows.

**They are not all held-out predictions.** That model trained on ~80% of these rows, so
most of this column is in-sample. The mean absolute error over the whole column (0.0289 eV)
is correspondingly optimistic. The honest generalisation figure is the production model's
**validation MAE of 0.0743 eV** on 12,695 molecules it never saw
(`04_modeling/production_model/result.json`).

There is no split column in this file, so in-sample and held-out rows cannot be told
apart from the CSV alone. Use `training_inputs/splits.json` if that distinction matters.

## Dropped columns

The compiled DFT table this was derived from also carried `timestamp`, `radical_name`,
`batch_id`, `radical_number`, `radical_type`, `S2`, and four cluster assignments. All are
collection bookkeeping or analysis scaffolding rather than data a reader needs, and are
omitted here.

`S2` was ⟨S²⟩, the spin-contamination diagnostic from the unrestricted DFT wavefunction
(0.75 exactly for a clean doublet). It was recorded for only 31,570 of the source rows,
spanning 0.7517–0.8497 with a median of 0.7556 — mild contamination throughout, nothing
that would flag a calculation as unusable.
