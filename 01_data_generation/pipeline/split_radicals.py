from pathlib import Path
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from functions import read_philicity, split_radicals, append_row_locked, split_radicals_heteroatom_h
from pathlib import Path
import csv
import fcntl
import os
from datetime import datetime
from functions import xyz_to_smiles
from functions import xyz_to_smiles, split_radicals, append_row_locked, get_next_radical_number_locked, append_rows_with_auto_radical_numbers_locked


def parse_args():

    p = argparse.ArgumentParser(
        description="split radical files after goat optimization",
    )

    p.add_argument("--goat_directory", required=True, help="directory to look for goat file")

    p.add_argument("--molname", required=True, help="molecule name")

    p.add_argument("--clusternum", required=True, help="cluster label")

    p.add_argument("--original_smiles", required=True, help="original smiles")

    p.add_argument("--batch_number", required=True, help="batch number")

    return p.parse_args()

def main():

    args = parse_args()

    goat_directory = args.goat_directory
    molname = args.molname
    clusternum = args.clusternum
    original_smiles = args.original_smiles
    batch_number = args.batch_number

    results_path = (
        "/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/batches/"
       f"batch_{batch_number}_radical_dataframe.csv"
    )

    hydrogen_indices = split_radicals_heteroatom_h(goat_directory, {"C": 1.25, "N": 1.25, "O": 1.05, "S": 1.50})

    rows = []
    seen_smiles = set()   # 🔑 track unique radicals per molecule

    for i in hydrogen_indices:
        radical_smiles = xyz_to_smiles(f"{goat_directory}/{i}H.xyz")

        if radical_smiles is None:
            continue

        # 🔒 skip symmetry-equivalent hydrogens
        if radical_smiles in seen_smiles:
            continue

        seen_smiles.add(radical_smiles)

        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "original_smiles": original_smiles,
            "molname": molname,
            "cluster_assignment": clusternum,
            "hydrogen_indices": ";".join(map(str, hydrogen_indices)),
            "specific_hydrogen": i,
            "radical_smiles": radical_smiles,
            # radical_number assigned atomically later
        })

    append_rows_with_auto_radical_numbers_locked(results_path, rows)

if __name__ == '__main__':

    main()