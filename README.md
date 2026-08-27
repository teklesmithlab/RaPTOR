# RaPTOR — Radical Philicity Prediction

Project directory for the RaPTOR paper: a D-MPNN that predicts the Parr electrophilicity
index of open-shell radicals from structure alone.

Philicity throughout is **ω = (I + A)² / (8(I − A))**, in eV, computed from DFT
ionization potential and electron affinity. The dataset spans carbon-, nitrogen-,
oxygen- and sulfur-centred radicals.

Live model: https://huggingface.co/spaces/mklavin/RaPTOR

---

## Headline numbers

| | |
|---|---|
| Dataset | 142,779 radicals (C 90,927 / S 20,511 / N 18,495 / O 12,846) |
| Split (train/val/test) | 128,731 / 12,695 / 1,353 |
| Validation MAE | **0.0743 eV** |
| Validation RMSE | 0.1269 eV |
| Inference cost | ~0.9 ms per molecule, CPU |

Production architecture: `depth=10`, `message_hidden_dim=1250`, `ffn_hidden_dim=750`,
`atom_ffn_hidden_dim=750`, `ffn/atom_ffn_num_layers=2`, `dropout=0.0`, `batch_size=64`,
`max_lr=3e-4`, 50 epochs, patience 20, ensemble size 1.

---

## Directory map

```
01_data_generation/    ORCA/xtb pipeline that produced the DFT dataset
02_data_preparation/   filtering, augmentation, preprocessing
03_dataset/            raptor_dataset.csv — the 142,779-radical dataset
04_modeling/           hyperparameter search, training, the trained models
```

Each directory has its own README describing its contents.

---

## Reproduction chain

```
01  functions.py + submit_batches.py + submit_philicities.py
        -> per-molecule ORCA jobs -> read_files.py
                |
02  compile_and_filter.py      -> the compiled DFT table
    preprocess_for_modeling.py -> cleaning, artifact rejection
                |
04  chemprop_all_prep.py       -> production_model/training_inputs/
    chemprop_grid_*_train.sh   -> production_model/best.pt   (4-stage grid search)
    chemprop_loeo_*            -> leave-one-element-out zero-shot generalisation
                |
03  build_raptor_dataset.py    -> raptor_dataset.csv
```

Note the last step is numbered 03 but runs last: the published dataset carries a
`predicted_philicity` column, so it can only be assembled once a model exists.

The intermediate tables between 02 and 04 are not shipped — the scripts read them by
absolute path from the original working tree. See **Deliberately not included**.

---

## Two things that silently corrupt predictions

Both are load-bearing and easy to lose in an edit:

1. **The training target is `log1p(philicity)`.** Model output must go through `expm1`
   to be read as eV.
2. **chemprop's `build_dataloader` sets `drop_last=True`** whenever
   `len(dataset) % batch_size == 1`, silently dropping the last molecule. Inference must
   pass `drop_last=False`.

The deployed `raptor_inference.py` in the Space repo handles both correctly and is the
reference implementation — see **Deployment** below.

---

## Deliberately not included

This directory is a curated publication set, not a mirror of the working tree.

| Excluded | Why | Where it lives |
|---|---|---|
| Raw per-molecule DFT output (~119 GB, 105k directories) | Far too large to publish inline; belongs in a data repository | `project_5/data_generation/molecules/` |
| CCSD(T) benchmark data and scripts | Submitted separately with the paper | `project_5/benchmarking/`, `project_5/data_analysis/benchmarking/` |
| Conformer-sensitivity raw DFT (~15 GB) | Supporting study, not part of this set | `project_5/benchmarking/conformer_analysis/` |
| Clustering, and the leave-one-cluster-out study built on it | Not part of the paper; the element-level generalisation study (LOEO) is the one reported | `project_5/data_analysis/*_cluster*.py`, `project_5/modeling/loco_cv_*` |
| Superseded model lineage (TensorFlow/nfp `v16`–`v37`) | Replaced by the chemprop model 2026-07-30 | `project_5/modeling/models/` |
| Hyperparameter-search trial checkpoints (~31 GB of `.ckpt`) | Only needed to resume training; winning configs recorded in `04_modeling/scripts/` | `project_5/modeling/chemprop_runs/` |
| Analysis and figure-generation scripts, and their derived tables | Figures are published in the paper itself; the code that draws them is not part of this set | `project_5/data_analysis/` |
| The web application | Published as its own Hugging Face Space repo — see **Deployment** below | `project_5/RaPTOR/` |
| `Avenir.ttc` | **Proprietary font — not redistributable** | `project_5/data_analysis/Avenir.ttc` |
| Space query logs | User-submitted structures from the public site; must not be published | `project_5/RaPTOR/query_logs/` (gitignored) |

---

## Known open issues

Carried forward so they are not rediscovered late:

- **Duplicate structures in the dataset.** 7,224 SMILES appear in more than one row;
  1,861 of those disagree with themselves by >0.05 eV (max 1.99 eV). This is a
  data-collection artifact, independent of any modelling choice, and is not corrected by
  any script here.

- **`predicted_philicity` is mostly in-sample.** It comes from the paper-split model,
  which trained on ~80% of these rows. Its whole-column MAE (0.0289 eV) is optimistic;
  the honest generalisation number is the production model's 0.0743 eV validation MAE.
  See `03_dataset/README.md`.

- **Sulfur is the weakest element.** Test R² = 0.818 versus 0.98 (C/N) and 0.94 (O).
  Absolute MAE (0.117 eV) is in line with N and O; the issue is that sulfur's philicity
  range is narrow (std 0.41 eV), so a predict-the-mean baseline already reaches 0.31 eV.

---

## Deployment

The model is served from a Hugging Face Space, which is its own git repository and is not
duplicated here:

- **Space:** https://huggingface.co/spaces/mklavin/RaPTOR
- **Remote:** `git@hf.co:spaces/mklavin/RaPTOR`

That repo holds `api.py`, `routes_ui.py`, `raptor_inference.py`, `mimis_drawing.py`,
`query_log.py`, `requirements.txt`, `Dockerfile`, `static/`, and `raptor_model/best.pt` —
the same checkpoint as `04_modeling/production_model/best.pt`
(md5 `7e259e8d3b7ef583fce8a6fd6545e891`).

---

## Licence

CC BY 4.0 (`LICENSE`), covering the code, the dataset and the model weights alike. Use,
adapt and redistribute freely, including commercially — the only condition is
attribution. For academic use, citing the RaPTOR paper satisfies that.

---

## Environment

Training and inference: `chemprop 2.2.3`, `torch 2.5.1`, `rdkit`, `pandas`, `numpy`.
The pinned inference stack is `requirements.txt` in the Space repo.

Scripts carry absolute paths into the original working tree
(`/insomnia001/depts/tekle_smith/users/MKL/project_5/...`). They are preserved verbatim
so the provenance of every number is auditable; adjust the path constants at the top of
each script to re-run elsewhere.
