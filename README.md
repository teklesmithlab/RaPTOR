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

CC BY 4.0 (`LICENSE`), covering the code, the dataset and the model weights alike. For academic use, please cite the RaPTOR paper.

---

## Environment

Training and inference: `chemprop 2.2.3`, `torch 2.5.1`, `rdkit`, `pandas`, `numpy`.
The pinned inference stack is `requirements.txt` in the Space repo.

Scripts carry absolute paths into the original working tree
(`/insomnia001/depts/tekle_smith/users/MKL/project_5/...`). They are preserved verbatim
so the provenance of every number is auditable; adjust the path constants at the top of
each script to re-run elsewhere.
