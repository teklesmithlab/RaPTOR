# Hyperparameter search, training schedule, and seed variance
### Source material for referee points 3-5. Generated 2026-07-30 from the run directories.

All values below were read directly from `chemprop_runs/*/configs.json`,
`*/config.toml`, `*/result.json`, and the saved checkpoints. Validation MAE/RMSE
are in eV, computed at the radical-center atom after undoing the log1p transform
(n_val = 12,695 for every run).

---

## IMPORTANT: two of the four searched axes were inert

The model is a `MolAtomBondMPNN` with **only the atom head trained**
(`output_columns = [None, ['atom_y'], None]`). The checkpoint contains
`mol_predictor: None` and no `mol_predictor.*` tensors in the state dict.

Consequently `--ffn-hidden-dim` and `--ffn-num-layers`, which configure the
*molecule-level* readout, build nothing. Proof: the four stage-3 runs that differ
only in `ffn_hidden_dim` (300/750/1500/2200) have **identical parameter counts**,
4,828,653, and identical tensor counts (18).

This has two consequences:

1. The search covered **72 distinct configurations across 105 runs**, not 105
   distinct configurations. Describing the FFN shape as "tuned" overstates it:
   only the *atom* FFN width and depth were ever varied.
2. The duplicated runs are **free seed replicates** — `--pytorch-seed` was never
   set, so `chemprop/cli/train.py:1757` calls `torch.seed()` and each run draws a
   fresh random seed with `deterministic = False`. This is what makes the variance
   table below possible without new compute.

---

## Table 1 — Staged grid search

"Staged" = sequential coarse-to-fine: each stage fixes the winner of the previous
stage and varies the next parameter group. Stages were not revisited.

| Stage | Parameters varied | Values searched | Configs | Selected |
|---|---|---|---|---|
| 1 | message-passing depth | 8, 9, 10, 11, 12, 13, 14, 15 | 48 (45 completed) | **10** |
|   | message hidden dim | 300, 500, 750, 1000, 1250, 1500 | | **1250** |
| 2 | max learning rate | 1e-4, 3e-4, 1e-3 | 12 | **3e-4** |
|   | batch size | 32, 64, 128, 256 | | **64** |
| 3 | atom FFN hidden dim | 300, 750, 1500 | 36 runs / **9 distinct** | **750** |
|   | dropout | 0.0, 0.1, 0.2 | | **0.0** |
|   | *(mol FFN hidden dim)* | *300, 750, 1500, 2200 — inert* | | *n/a* |
| 4 | atom FFN num layers | 1, 2, 3 | 9 runs / **3 distinct** | **2** |
|   | *(mol FFN num layers)* | *1, 2, 3 — inert* | | *n/a* |

Held fixed during stage 1 at Chemprop defaults: max_lr 1e-3, batch size 64.
Held fixed from stage 2 onward: depth 10, message hidden dim 1250.
Every stage: 50 epochs, patience 20, ensemble size 1, aggregation `norm` (norm 100),
ReLU, `multi-hot-atom-featurizer-mode V2`.

**Stage-2 caveat:** `lr 3e-4 / bs 64` and `lr 1e-4 / bs 32` tie exactly at 0.07552 eV.
The selection between them was arbitrary.

---

## Table 2 — Training schedule

| Item | Value |
|---|---|
| Optimizer | Adam (`chemprop/models/model.py:209`) |
| LR schedule | Noam-like: linear warmup then exponential decay |
| Warmup | 2 epochs |
| Initial / max / final LR | 1e-4 / 3e-4 / 1e-4 |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | on `val_loss`, patience 20, min_delta 0.0 |
| Loss | MSE on the standardized log1p target (`predictors.py:162`) |
| Target transform | log1p, then Chemprop's internal standardization |
| Ensemble size | 1 |
| Data seed | 0 (split only) |
| Torch seed | **unset** — random per run |

**Convergence caveat:** best epochs across the five replicate runs were 46, 45, 46,
47, and 13, against a 50-epoch ceiling. Four of five peaked in the last 10% of the
budget without early stopping ever firing, so the epoch budget was binding and the
model was probably still improving. A referee may well ask about this; raising the
ceiling to 100-150 epochs and reporting where it actually converges is the cleanest
answer.

---

## Table 3 — Seed variance (referee point 5)

Five independent runs of the **identical** selected configuration — byte-identical
`config.toml` apart from `output-dir`, same data, same split, differing only in the
random torch seed. Four come from the inert `ffn_hidden_dim` axis of stage 3, the
fifth is `grid_layers/grid_fl2_afl2`.

| Run | val MAE (eV) | val RMSE (eV) | best epoch |
|---|---|---|---|
| grid_ffn750_affn750_do0p0 *(the deployed model)* | 0.07426 | 0.12692 | 45 |
| grid_ffn2200_affn750_do0p0 | 0.07442 | 0.12698 | 47 |
| grid_ffn300_affn750_do0p0 | 0.07445 | 0.12749 | 46 |
| grid_ffn1500_affn750_do0p0 | 0.07481 | 0.12762 | 46 |
| grid_fl2_afl2 | 0.07751 | 0.12886 | 13 |
| **mean ± std (n=5)** | **0.0751 ± 0.0014** | **0.1276 ± 0.0008** | |

**The currently reported 0.0743 eV is the minimum of five runs, not a typical run.**
The honest single-model number is 0.0751 ± 0.0014 eV.

### Per-cell variance across the whole of stage 3

Each cell is 4 seeds (the inert mol-FFN axis):

| atom FFN dim | dropout | mean MAE | std | spread |
|---|---|---|---|---|
| 300 | 0.0 | 0.07744 | 0.00163 | 0.00357 |
| 300 | 0.1 | 0.07691 | 0.00118 | 0.00288 |
| 300 | 0.2 | 0.07738 | 0.00146 | 0.00278 |
| **750** | **0.0** | **0.07448** | **0.00023** | 0.00055 |
| 750 | 0.1 | 0.07823 | 0.00133 | 0.00271 |
| 750 | 0.2 | 0.07728 | 0.00144 | 0.00345 |
| 1500 | 0.0 | 0.07508 | 0.00040 | 0.00081 |
| 1500 | 0.1 | 0.07676 | 0.00102 | 0.00216 |
| 1500 | 0.2 | 0.07763 | 0.00169 | 0.00357 |

### What survives the noise, and what does not

Typical within-cell std is 0.0010-0.0019 eV. Against that:

- **Real:** dropout 0.0 beats 0.1/0.2 (gap ~0.003 eV, roughly 2x the std). This
  supports the existing "dropout 0.0 won" conclusion.
- **Real:** atom FFN 750 beats 300 (0.0745 vs 0.0774).
- **Not resolved:** atom FFN 750 vs 1500 (0.0745 vs 0.0751, gap 0.0006 — well inside
  one std).
- **Not resolved:** the entire stage-4 layer grid. Cell means are 0.07545 (1 layer),
  0.07682 (2 layers), 0.07608 (3 layers), all within ~1 std of each other. Note the
  deployed model uses 2 layers, which has the *worst* cell mean of the three.
- **Not resolved:** the top of stage 2 (0.07552-0.07585 across four configs).

---

## Suggested revision to the methods paragraph

> Radical philicity was modeled with Chemprop (v2.2.3). Hyperparameters were selected
> by a four-stage grid search on validation MAE, each stage fixing the previous
> stage's selection: (i) message-passing depth and hidden dimension, (ii) learning
> rate and batch size, (iii) atom-level FFN width and dropout, (iv) atom-level FFN
> depth — 72 distinct configurations in total (Table S1). The selected model uses a
> message-passing depth of 10, message hidden dimension 1250, and an atom-level
> readout FFN with two 750-unit hidden layers (`--atom-ffn-num-layers 2`), dropout
> 0.0, and ReLU activations; 4.83M parameters. Training minimized mean squared error
> on the log1p-transformed philicity target, which Chemprop additionally standardizes
> before optimization, using Adam with a Noam-like schedule (2 warmup epochs; initial,
> maximum, and final learning rates 1e-4, 3e-4, 1e-4), batch size 64, for at most 50
> epochs with early stopping on validation loss (patience 20). All MAE/RMSE/R2 values
> are computed after undoing both transforms. Because Chemprop's torch seed was left
> unset, we report validation performance as mean +/- s.d. over five independent runs
> of the selected configuration: MAE 0.0751 +/- 0.0014 eV, RMSE 0.1276 +/- 0.0008 eV
> (n_val = 12,695).

### Corrections this makes to the current text

1. **"3-layer/750-unit FFN readout"** -> the flag value is `--atom-ffn-num-layers 2`,
   which builds two 750-unit hidden layers (three linear transformations:
   1250->750->750->1). Writing "3-layer" is defensible as a count of linear maps, but
   a referee reproducing from it will set `--atom-ffn-num-layers 3` and get a
   different model. Give the flag value, or state both.
2. **"FFN shape"** -> only the *atom* FFN was tuned. The molecule-level FFN does not
   exist in this model.
3. **Add the missing selected values**: max_lr 3e-4, batch size 64.
4. **0.0743 -> 0.0751 +/- 0.0014**, or state explicitly that 0.0743 is the best of
   five runs.

---

## Recommended follow-up before submission

The five replicates above are legitimate but were not *designed* as replicates. The
cleanest version for a referee is 3-5 runs of the selected configuration with
explicit `--pytorch-seed 0..4`, which also makes them reproducible. Each run is
roughly a few GPU-hours on an A6000, so the whole set is one overnight array job.

Worth deciding at the same time: whether to raise the 50-epoch ceiling, given four
of five runs peaked at epoch 45-47.
