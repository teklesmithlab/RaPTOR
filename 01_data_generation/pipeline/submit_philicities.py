import pandas as pd
from functions import create_sh_file, create_DFT_inp_file, run_sh_file
from pathlib import Path
import argparse
import numpy as np
from pathlib import Path
from functions import read_philicity
from pathlib import Path
import csv
import fcntl
import os
from datetime import datetime
from functions import xyz_to_smiles

def parse_args():
    p = argparse.ArgumentParser(description="submitting DFT files for philicity calculations")
    p.add_argument("--dataframe_directory", required=True, help="CSV with radical rows")
    p.add_argument("--root_directory", required=True, help="root directory on cluster")
    p.add_argument("--secondary_directory", required=True, help="molname to operate on (e.g. p5_0)")
    p.add_argument("--batch_number", required=True, help="batch number")
    p.add_argument("--account", default="tekle_smith", help="SLURM account to charge")
    p.add_argument("--partition", default="short", help="SLURM partition (short caps at 12h, 89 nodes)")
    return p.parse_args()

def main():

    args = parse_args()

    dataframe_directory = args.dataframe_directory
    root_directory = args.root_directory
    secondary_directory = args.secondary_directory  # molname
    batch_number = args.batch_number
    account = args.account
    partition = args.partition

    radical_df = pd.read_csv(dataframe_directory)

    rows = radical_df[radical_df["molname"] == secondary_directory].copy()

    if rows.empty:
        print(f"No rows found for molname={secondary_directory}")
        return

    if "specific_hydrogen" in rows.columns:
        rows["specific_hydrogen"] = rows["specific_hydrogen"].astype(int)
        rows = rows.sort_values(["specific_hydrogen"])

    if "radical_number" in rows.columns:
        rows["radical_number"] = rows["radical_number"].astype(int)
    
    hydrogen_readfiles = []

    for _, row in rows.iterrows():

        h = int(row["specific_hydrogen"])
        radnum = int(row["radical_number"]) if "radical_number" in row and pd.notna(row["radical_number"]) else None

        # create read_files sh file
        create_sh_file(root_directory, secondary_directory, f"read_files_{h}H.sh", "", "readfiles", radname=secondary_directory, clusternum=row["cluster_assignment"], smiles=row["radical_smiles"], original_smiles=row["original_smiles"], molname=row["molname"], hydrogen_indices=row["hydrogen_indices"], specific_hydrogen=h, radical_number=radnum, batch_number=batch_number, account=account, partition=partition)

        # create geometry optimization sh file
        create_sh_file(root_directory, secondary_directory, f"geom_{h}H.sh", f"{h}H.xyz", 'fairchem', account=account, partition=partition)

        # create DFT input and sh files for radical calculation
        create_DFT_inp_file(root_directory, secondary_directory, f"DFT_elec_radical_{h}H.inp", f"geom_{h}H.xyz", 0, 2)
        create_sh_file(root_directory, secondary_directory, f"DFT_elec_radical_{h}H.sh", f"DFT_elec_radical_{h}H.inp", '$ORCA_PATH/orca', account=account, partition=partition)

        # create DFT input and sh files for anion calculation
        create_DFT_inp_file(root_directory, secondary_directory, f"DFT_elec_anion_{h}H.inp", f"geom_{h}H.xyz", -1, 1)
        create_sh_file(root_directory, secondary_directory, f"DFT_elec_anion_{h}H.sh", f"DFT_elec_anion_{h}H.inp", '$ORCA_PATH/orca', account=account, partition=partition)

        # create DFT input and sh files for cation calculation
        create_DFT_inp_file(root_directory, secondary_directory, f"DFT_elec_cation_{h}H.inp", f"geom_{h}H.xyz", 1, 1)
        create_sh_file(root_directory, secondary_directory, f"DFT_elec_cation_{h}H.sh", f"DFT_elec_cation_{h}H.inp", '$ORCA_PATH/orca', account=account, partition=partition)

        job_ID_geom = run_sh_file(root_directory, secondary_directory, f"geom_{h}H.sh")
        job_ID_radical = run_sh_file(root_directory, secondary_directory, f"DFT_elec_radical_{h}H.sh", job_ID_geom)
        job_ID_anion = run_sh_file(root_directory, secondary_directory, f"DFT_elec_anion_{h}H.sh", job_ID_geom)
        job_ID_cation = run_sh_file(root_directory, secondary_directory, f"DFT_elec_cation_{h}H.sh", job_ID_geom)
        job_ID_readfiles = run_sh_file(root_directory, secondary_directory, f"read_files_{h}H.sh", dependency_job_id=[job_ID_radical, job_ID_anion, job_ID_cation]) 
        hydrogen_readfiles.append(job_ID_readfiles)
    
    create_sh_file(root_directory, secondary_directory, f"delete_files.sh", "", "delete_files", molname=secondary_directory, account=account, partition=partition)
    run_sh_file(root_directory, secondary_directory, f"delete_files.sh", dependency_job_id=hydrogen_readfiles)

    print(f"Submitted read_files jobs for {secondary_directory}: {len(rows)} hydrogens")

if __name__ == '__main__':

    main()