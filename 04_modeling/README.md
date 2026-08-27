# 04 — Modeling

## production_model/

The model served by the RaPTOR website. `best.pt` here is bit-identical (md5
`7e259e8d3b7ef583fce8a6fd6545e891`) to the checkpoint deployed in the Hugging Face Space
repo (`git@hf.co:spaces/mklavin/RaPTOR`).

| File | Contents |
|---|---|
| `best.pt` | the trained chemprop model (19 MB) |
| `config.toml` | full training configuration |
| `result.json` | val MAE 0.074258 eV, val RMSE 0.126921 eV, n_val 12,695 |
| `val_preds.csv` | predictions on all 12,695 validation molecules |
| `test_predictions.csv` | predictions on the 1,353 test molecules |
| `training_inputs/` | the chemprop-format inputs (see below) |

Architecture: `depth=10`, `message_hidden_dim=1250`, `ffn_hidden_dim=750`,
`atom_ffn_hidden_dim=750`, `ffn/atom_ffn_num_layers=2`, `dropout=0.0`, `batch_size=64`,
`max_lr=3e-4`, 50 epochs, patience 20, ensemble size 1. Trained as cell
`grid_ffn750_affn750_do0p0` of the stage-3 grid.

### training_inputs/

What chemprop actually consumed — kept because the published dataset alone is not in
chemprop's input format:

- `all_radicals.csv` — `smiles`, `atom_y`, `split`, `radical_type`. `atom_y` is a
  per-atom list holding `log1p(philicity)` at the radical centre and `nan` elsewhere.
- `all_radicals_atom_features.npz` — per-atom features, row-aligned to the CSV.
- `splits.json` — index lists: 128,731 train / 12,695 val / 1,353 test.

All three are row-order dependent on each other **and on
`03_dataset/raptor_dataset.csv`**, which is in the same order.

> The equivalent file in the original working tree
> (`project_5/modeling/chemprop_data_all/all_radicals.csv`) was regenerated on
> 2026-07-20, five days after this model trained, with a reshuffled row order and a fresh
> split column. Same molecules, but its indices no longer align with `splits.json`. The
> copy here is the pre-regeneration snapshot and is the one that reproduces the model.

## paper_split_model/

A second model trained on a stratified 80/10/10 split, used for the parity and
error-distribution panels of the manuscript figure and the source of the
`predicted_philicity` column in `03_dataset/`. `all_preds.csv` covers the full dataset
under that split — note it is in the reshuffled row order, not the production order.

Kept separate from the production model deliberately: different splits, and their
numbers are not interchangeable.

## scripts/

The hyperparameter search ran in four sequential stages, each fixing the winners of the
last:

| Stage | Searched | Outcome |
|---|---|---|
| `chemprop_grid_search_*` | depth × message_hidden_dim | depth=10, msg=1250 |
| `chemprop_grid_lr_bs_*` | max_lr × batch_size | 3e-4, 64 |
| `chemprop_grid_ffn_*` | ffn × atom_ffn × dropout (36 cells) | 750 / 750 / 0.0 |
| `chemprop_grid_layers_*` | layer counts | 2 / 2 |

Dropout 0.0 won its grid outright and again on a stricter split.

> `chemprop_all_prep.py` is preserved verbatim for provenance and reads a clustered
> input file that is not shipped in this directory. Update its input path before reusing
> it to build new training data.

## scripts/validation/

**Leave-one-element-out (LOEO)** — `chemprop_loeo_prep.py`, `chemprop_loeo_train.sh`,
`loeo_evaluate.py`, `loeo_zeroshot_export.py`. Drops C, N, O or S from training entirely
and tests zero-shot on the held-out element. Because carbon is 61% of the data, the
carbon fold ships with a `sizematch_carbon` control that trains on all four elements
downsampled to the no-carbon row count — so the carbon ablation is not confounded by
training-set size. The heteroatom folds need no such control (each is only 9-16% of the
data) and are genuine zero-shot tests. This is the study behind panels E-H of the
manuscript figure.

**Learning curve** — `chemprop_kcurve_*`, held-out performance versus training-set size.

Trial checkpoints and the aggregated result tables are not included.
