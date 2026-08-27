import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from functions import read_philicity, append_row_locked, find_imaginary_nodes, check_normal_termination, read_spin_contamination
from pathlib import Path
import csv
import fcntl
import os
from datetime import datetime

def parse_args():

    p = argparse.ArgumentParser(
        description="submitting DFT files and reading when the calculations are done to extract philicities",
    )

    p.add_argument("--radname", required=True, help="radical name to read philicity for")

    p.add_argument("--clusternum", required=True, help="cluster label")

    p.add_argument("--smiles", required=True, help="radical smiles")

    p.add_argument("--original_smiles", required=True, help="original smiles")

    p.add_argument("--molname", required=True, help="molecule name, secondary file location")

    p.add_argument("--hydrogen_indices", required=True, help="list of hydrogens on the molecule")

    p.add_argument("--specific_hydrogen", required=True, help="hydrogen to run the calculation for")

    p.add_argument("--radical_number", required=True, help="index of the radical")

    p.add_argument("--batch_number", required=True, help="batch number")

    return p.parse_args()

def main():

    args = parse_args()

    radname = args.radname
    clusternum = args.clusternum
    smiles = args.smiles
    original_smiles = args.original_smiles
    molname = args.molname
    hydrogen_indices = args.hydrogen_indices
    specific_hydrogen = args.specific_hydrogen
    radical_number = args.radical_number
    batch_number = args.batch_number

    mol_dir = f'/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/molecules/{molname}/'
    anion_file  = f'{mol_dir}DFT_elec_anion_{specific_hydrogen}H.out'
    cation_file = f'{mol_dir}DFT_elec_cation_{specific_hydrogen}H.out'
    radical_file = f'{mol_dir}DFT_elec_radical_{specific_hydrogen}H.out'

    if not find_imaginary_nodes(mol_dir, specific_hydrogen):

        # Skip conformers where any ORCA job did not finish cleanly
        if not all(check_normal_termination(f) for f in [anion_file, cation_file, radical_file]):
            print(f"Skipping {radname} H={specific_hydrogen}: ORCA did not terminate normally for at least one job")
            return

        # Skip conformers with significant spin contamination (S**2 > 0.85 for a doublet).
        # Pi-conjugated N/C radicals (benzylic, allylic, pyridyl) routinely reach
        # 0.80-0.84 due to spin delocalization — that is normal DFT behaviour, not
        # an error. The 0.85 threshold targets genuinely contaminated cases (aryl
        # sigma-radicals at ~1.77) while preserving legitimate conjugated radicals.
        s2 = read_spin_contamination(radical_file)
        if s2 is not None and s2 > 0.85:
            print(f"Skipping {radname} H={specific_hydrogen}: spin contamination S**2={s2:.4f}")
            return

        I, A, philicity = read_philicity(mol_dir, specific_hydrogen)

        # Carbon radicals legitimately have A ≤ 0 (alkyl radicals don't bind
        # electrons). For oxygen-centered radicals however, A < 0 is unphysical —
        # alkoxy/acyloxy radicals should have A ≈ 1–4 eV. A near-zero or negative
        # A on an O-radical SMILES indicates FairChem drove beta-scission.
        # In canonical SMILES, '[O]' (no charge, no explicit H) is unambiguously
        # an oxygen radical centre, so this check is safe and targeted.
        if '[O]' in smiles and A < 0:
            print(f"Skipping {radname} H={specific_hydrogen}: O-radical with negative EA A={A:.4f} eV (beta-scission artifact)")
            return

        results_path = Path(f"/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/batches/batch_{batch_number}_philicities.csv")

        row = {
            "radical_number": radical_number,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "original_smiles": original_smiles,
            "radical_name": radname,
            "cluster_assignment": clusternum,
            "radical_smiles": smiles,
            "philicity": philicity,
            "I": I,
            "A": A,
            "S2": s2,
        }

        append_row_locked(results_path, row)

if __name__ == '__main__':

    main()