# 02 — Data preparation

Everything between raw DFT output and the modelling dataset.

| File | Role |
|---|---|
| `preprocess_for_collection.py` | Prepares candidate structures for DFT submission. |
| `filtering.py` | Structural and validity filters. |
| `augment.py` | Dataset augmentation. |
| `preprocess_for_modeling.py` | Main cleaning pass: validity, radical-centre checks, DFT-artifact rejection. |
| `compile_and_filter.py` | Compiles per-batch philicity files into a single table. |

Output feeds `03_dataset/raptor_dataset.csv`.

## Using these

These are **function libraries, not runnable scripts** — import from them and pass your
own paths. The one-off driver blocks that used to sit at the bottom of `augment.py`,
`filtering.py` and `preprocess_for_collection.py` were removed: they pointed at working
directories that no longer exist and were never a documented entry point.

The section is also **not self-contained.** The intermediate tables these read (the raw
per-batch philicity files, the compiled DFT table, and the DFT-artifact rejection list
`worst_performing_radicals_v21_localization.csv`) are not shipped here — publishing them
would mean publishing the ~119 GB of raw DFT output they derive from. The scripts run
against the original working tree, where those inputs are all present. They are included
to document exactly how the dataset was cleaned, not as a turnkey pipeline.
