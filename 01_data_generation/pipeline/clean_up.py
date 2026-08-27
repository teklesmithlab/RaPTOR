from pathlib import Path
from rdkit import Chem
import pandas as pd
from typing import Dict

# Never delete these regardless of extension/name — analysis code, tabulated
# results, and the goat conformer-ensemble outputs (not just goat.out).
KEEP_SUFFIXES = {".py", ".csv", ".md", ".json", ".ipynb"}
KEEP_NAME_SUBSTRINGS = ("goat.finalensemble", "goat.globalminimum", "fairchem")

def delete_non_out_files(root_dir, dry_run=True):

    root_dir = Path(root_dir)

    if not root_dir.exists() or not root_dir.is_dir():
        return

    deleted = 0

    for path in root_dir.rglob("*"):
        # skip broken symlinks / race conditions
        try:
            if not path.is_file():
                continue
        except FileNotFoundError:
            continue

        if path.suffix in KEEP_SUFFIXES or any(s in path.name for s in KEEP_NAME_SUBSTRINGS):
            continue

        delete_file = (
            ("slurm" in path.name)
            or ("atom" in path.name)
            or ("delete" in path.name)
            or ("read" in path.name)
            or ("submit" in path.name)
            or (path.suffix != ".out")
            or ("goat.0." in path.name)
            or ("goat.1." in path.name)
            or ("goat.2." in path.name)
            or ("goat.3." in path.name)
            or ("goat.4." in path.name)
            or (".xyz" in path.name)
            or (".sh" in path.name)
            or (".inp" in path.name)
        )

        if delete_file:
            if dry_run:
                print(f"[DRY RUN] Would delete: {path}")
            else:
                try:
                    path.unlink()
                    deleted += 1
                except FileNotFoundError:
                    # file vanished between listing and delete — fine
                    continue

    if not dry_run:
        print(f"Deleted {deleted} files from {root_dir}")

if __name__ == '__main__':

    delete_non_out_files('/insomnia001/depts/tekle_smith/users/MKL/project_5/benchmarking/', dry_run=False)