"""
Chunk the cleaned amine list into batch_NNN.csv files -- same name,clusternum,smiles
format and same batch_NNN.csv naming as the existing radical batches, just
continuing the batch-number sequence where the existing batches left off
(last existing batch is batch_2471.csv, so amine batches start at 2472) so
there's a single global batch-number counter across functional groups.

clusternum is a placeholder (repeated constant, 0) here: the fingerprint-
clustering step used to label existing batches (project_5/fingerprints/
cluster_data.py) isn't available for this amine set. clusternum is only a
bookkeeping label carried through to the results CSV -- it isn't used in the
DFT calculation itself -- so this doesn't affect the philicity values, only
downstream diversity analysis grouping.
"""

from pathlib import Path

import pandas as pd

INPUT_FILE = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/nitrogen_radicals/all_amines_filtered.csv")
OUTPUT_DIR = Path("/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/batches")
BATCH_SIZE = 100
NAME_PREFIX = "p5_"  # same global molecule-name counter as the existing radical batches
START_BATCH_NUMBER = 2472  # next free batch number after the existing batch_2471.csv
START_NAME_INDEX = 247200  # next free molecule index in the shared p5_ counter


def main():
    df = pd.read_csv(INPUT_FILE)

    out = pd.DataFrame({
        "name": [f"{NAME_PREFIX}{START_NAME_INDEX + i}" for i in range(len(df))],
        "clusternum": 0,
        "smiles": df["SMILES"],
    })

    n_batches = 0
    for start in range(0, len(out), BATCH_SIZE):
        batch_number = START_BATCH_NUMBER + start // BATCH_SIZE
        chunk = out.iloc[start:start + BATCH_SIZE]
        out_path = OUTPUT_DIR / f"batch_{batch_number:03d}.csv"
        chunk.to_csv(out_path, index=False)
        n_batches += 1

    print(f"wrote {n_batches} batches ({len(out)} molecules) to {OUTPUT_DIR}, "
          f"batch_{START_BATCH_NUMBER:03d}.csv through batch_{START_BATCH_NUMBER + n_batches - 1:03d}.csv")


if __name__ == '__main__':
    main()
