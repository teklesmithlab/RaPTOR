"""
Analog of submit_philicities.py for amine radical cations.

There is no per-hydrogen loop here: the radical cation is the intact neutral
skeleton at charge=+1, so one molecule -> one FairChem geometry optimization
-> three DFT single points (neutral / radical cation / dication) -> one
read_files call. Compare to submit_philicities.py, which loops over every
abstractable hydrogen produced by split_radicals.py.
"""

import argparse
import shutil
from pathlib import Path

from functions import create_DFT_inp_file, create_sh_file, run_sh_file
from cation_functions import (
    RADICAL_CATION_STEM,
    create_fairchem_cation_sh_file,
    create_readfiles_cation_sh_file,
)


def parse_args():
    p = argparse.ArgumentParser(description="submitting DFT files for radical-cation philicity calculations")
    p.add_argument("--root_directory", required=True, help="root directory on cluster")
    p.add_argument("--secondary_directory", required=True, help="molname to operate on (e.g. ac_0)")
    p.add_argument("--clusternum", required=True, help="cluster label")
    p.add_argument("--original_smiles", required=True, help="original (neutral) smiles")
    p.add_argument("--batch_number", required=True, help="batch number")
    p.add_argument("--account", default="tekle_smith", help="SLURM account to charge")
    return p.parse_args()


def main():
    args = parse_args()

    root_directory = args.root_directory
    secondary_directory = args.secondary_directory
    clusternum = args.clusternum
    original_smiles = args.original_smiles
    batch_number = args.batch_number
    account = args.account

    workdir = Path(root_directory) / secondary_directory
    goat_min = workdir / "goat.globalminimum.xyz"
    radicalcation_xyz = workdir / f"{RADICAL_CATION_STEM}.xyz"

    if not goat_min.exists():
        print(f"No GOAT global minimum found for {secondary_directory}, skipping.")
        return

    # keep the neutral-optimized skeleton geometry intact -- no hydrogen removed
    shutil.copyfile(goat_min, radicalcation_xyz)

    geom_xyz = f"geom_{RADICAL_CATION_STEM}.xyz"

    # FairChem geometry optimization of the radical cation itself: charge +1, doublet
    create_fairchem_cation_sh_file(root_directory, secondary_directory, "fairchem_cation.sh",
                                    f"{RADICAL_CATION_STEM}.xyz", charge=1, spinmult=2, account=account)

    # three DFT single points at the optimized radical-cation geometry:
    # neutral (0,1) and dication (2,1) flank the radical cation (1,2) itself
    create_DFT_inp_file(root_directory, secondary_directory, "DFT_elec_neutral.inp", geom_xyz, 0, 1)
    create_sh_file(root_directory, secondary_directory, "DFT_elec_neutral.sh", "DFT_elec_neutral.inp", '$ORCA_PATH/orca', account=account)

    create_DFT_inp_file(root_directory, secondary_directory, "DFT_elec_radicalcation.inp", geom_xyz, 1, 2)
    create_sh_file(root_directory, secondary_directory, "DFT_elec_radicalcation.sh", "DFT_elec_radicalcation.inp", '$ORCA_PATH/orca', account=account)

    create_DFT_inp_file(root_directory, secondary_directory, "DFT_elec_dication.inp", geom_xyz, 2, 1)
    create_sh_file(root_directory, secondary_directory, "DFT_elec_dication.sh", "DFT_elec_dication.inp", '$ORCA_PATH/orca', account=account)

    create_readfiles_cation_sh_file(root_directory, secondary_directory, "read_files_cation.sh",
                                     molname=secondary_directory, clusternum=clusternum,
                                     original_smiles=original_smiles,
                                     batch_number=batch_number, account=account)

    job_ID_geom = run_sh_file(root_directory, secondary_directory, "fairchem_cation.sh")
    job_ID_neutral = run_sh_file(root_directory, secondary_directory, "DFT_elec_neutral.sh", job_ID_geom)
    job_ID_radicalcation = run_sh_file(root_directory, secondary_directory, "DFT_elec_radicalcation.sh", job_ID_geom)
    job_ID_dication = run_sh_file(root_directory, secondary_directory, "DFT_elec_dication.sh", job_ID_geom)
    job_ID_readfiles = run_sh_file(root_directory, secondary_directory, "read_files_cation.sh",
                                    dependency_job_id=[job_ID_neutral, job_ID_radicalcation, job_ID_dication])

    print(f"Submitted radical-cation jobs for {secondary_directory}, read_files job: {job_ID_readfiles}")


if __name__ == '__main__':
    main()
