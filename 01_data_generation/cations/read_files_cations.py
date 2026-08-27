"""
Analog of read_files.py for amine radical cations: one row per molecule
(no specific_hydrogen, since there's exactly one radical cation per parent
amine) computed from the neutral / radical-cation / dication DFT single
points instead of anion / radical / cation.

Output schema matches the existing batch_NNN_philicities.csv files exactly
(radical_number, timestamp, original_smiles, radical_name, cluster_assignment,
radical_smiles, philicity, I, A, S2) and is written to the same
batch_NNN_philicities.csv filename, so combine_finished_batches can pick up
cation batches the same way it does the neutral-radical ones.
"""

import argparse
from datetime import datetime
from pathlib import Path

from functions import (
    append_rows_with_auto_radical_numbers_locked,
    check_normal_termination,
    read_spin_contamination,
)
from cation_functions import (
    find_imaginary_nodes_cation,
    neutral_to_radical_cation_smiles,
    read_philicity_cation,
    RADICAL_CATION_STEM,
)


def parse_args():
    p = argparse.ArgumentParser(description="reading radical-cation DFT results to extract philicities")
    p.add_argument("--molname", required=True, help="molecule name, secondary file location")
    p.add_argument("--clusternum", required=True, help="cluster label")
    p.add_argument("--original_smiles", required=True, help="original (neutral) smiles")
    p.add_argument("--batch_number", required=True, help="batch number")
    return p.parse_args()


def main():
    args = parse_args()

    molname = args.molname
    clusternum = args.clusternum
    original_smiles = args.original_smiles
    batch_number = args.batch_number

    mol_dir = f'/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/molecules/{molname}/'
    neutral_file = f'{mol_dir}DFT_elec_neutral.out'
    radicalcation_file = f'{mol_dir}DFT_elec_radicalcation.out'
    dication_file = f'{mol_dir}DFT_elec_dication.out'

    if find_imaginary_nodes_cation(mol_dir):
        print(f"Skipping {molname}: imaginary modes present after FairChem optimization")
        return

    if not all(check_normal_termination(f) for f in [neutral_file, radicalcation_file, dication_file]):
        print(f"Skipping {molname}: ORCA did not terminate normally for at least one job")
        return

    # same spin-contamination gate as read_files.py, applied to the radical
    # cation itself (the doublet in this ladder, analogous to the old
    # neutral radical single point)
    s2 = read_spin_contamination(radicalcation_file)
    if s2 is not None and s2 > 0.85:
        print(f"Skipping {molname}: spin contamination S**2={s2:.4f}")
        return

    I, A, philicity = read_philicity_cation(mol_dir)

    # Derive the radical-cation SMILES from original_smiles (neutral amine) by
    # adding +1 formal charge to the amine nitrogen. xyz_to_smiles/obabel cannot
    # be used here because standard xyz files carry no charge/spin info, so obabel
    # returns a neutral SMILES with GetNumRadicalElectrons()=0 everywhere, causing
    # every cation row to be silently dropped downstream.
    radical_smiles = neutral_to_radical_cation_smiles(original_smiles, mol_dir=mol_dir)
    if radical_smiles is None:
        print(f"Skipping {molname}: no amine N found in '{original_smiles}'")
        return

    results_path = Path(f"/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/batches/batch_{batch_number}_philicities.csv")

    row = {
        "radical_number": None,  # overwritten with the next atomic value by append_rows_with_auto_radical_numbers_locked
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "original_smiles": original_smiles,
        "radical_name": molname,
        "cluster_assignment": clusternum,
        "radical_smiles": radical_smiles,
        "philicity": philicity,
        "I": I,
        "A": A,
        "S2": s2,
    }

    append_rows_with_auto_radical_numbers_locked(results_path, [row])


if __name__ == '__main__':
    main()
