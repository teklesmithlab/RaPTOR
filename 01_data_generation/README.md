# 01 — Data generation

The DFT pipeline that produced the philicity dataset. For each radical it optimises the
neutral geometry, then runs single-point calculations on the neutral, cation and anion
to obtain I and A, and derives ω = (I + A)² / (8(I − A)).

## pipeline/

| File | Role |
|---|---|
| `functions.py` | Core library — job construction, ORCA/xtb input generation, philicity calculation. `create_sh_file` emits the per-molecule SLURM chain. |
| `submit_batches.py` | Submits geometry-optimisation batches; self-resubmitting chain. |
| `submit_philicities.py` | Submits the I/A single-point stage. |
| `read_files.py` | Parses ORCA output back into philicity values. |
| `compute_xtb_features.py` | xtb-derived scalar descriptors. |
| `split_radicals.py`, `prepare_augmentation.py` | Input-set construction. |
| `clean_up.py`, `delete_files.py` | Per-molecule scratch cleanup (keep-list based; removes ORCA/xtb scratch, retains every `.out` and non-junk `.xyz`). |

## cations/

Parallel path for cationic species — same structure, separate submission chain.

## launchers/

The `sbatch` entry points. All pipeline jobs pin `--partition=short` (12 h cap);
walltime is `0-2:00` per molecule.
