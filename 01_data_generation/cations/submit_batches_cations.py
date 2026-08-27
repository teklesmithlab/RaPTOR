"""
Analog of submit_batches.py for amine radical cations.

Per molecule: xtb pre-opt -> GOAT conformer search (both still at neutral
charge/multiplicity -- we want the best neutral starting geometry) -> submit
the radical-cation pipeline (FairChem opt at charge +1 + 3 DFT single points
+ read_files_cations.py). There is no split_radicals.py step: a radical
cation is the intact neutral skeleton ionized, not a hydrogen-abstraction
product, so there's nothing to enumerate per molecule.
"""

import argparse

import pandas as pd
from functions import convert_SMILES_to_xyzs, create_sh_file, create_goat_inp_file, run_sh_file
from cation_functions import create_submit_cation_pipeline_sh_file, create_submit_batches_cations_sh_file


def parse_args():
    p = argparse.ArgumentParser(description="submit amine radical-cation batch for DFT data generation")

    p.add_argument("--root_directory", required=True, help="root file directory")
    p.add_argument("--batch_number", required=True, help="batch to run")
    p.add_argument("--account", default="tekle_smith", help="SLURM account to charge")
    p.add_argument("--chain_id", default="A", help="unique chain identifier (A or B)")
    p.add_argument("--max_batch", type=int, default=3000, help="stop chain after this batch number")

    return p.parse_args()


def main():
    args = parse_args()

    root_directory = args.root_directory
    batch_number = args.batch_number
    account = args.account
    chain_id = args.chain_id
    max_batch = args.max_batch

    batch_directory = f"{root_directory}/batches"
    molecules_directory = f"{root_directory}/molecules"

    batch_number = int(batch_number)
    smiles_file_name = f'batch_{batch_number:03d}.csv'

    end_number = convert_SMILES_to_xyzs(root_directory, smiles_file_name, 'ac_')
    print(end_number)

    smiles_dataframe = pd.read_csv(f"{batch_directory}/{smiles_file_name}")

    done_running = []

    for i in smiles_dataframe['name']:

        smiles_dataframe1 = smiles_dataframe.set_index("name")
        row = smiles_dataframe1.loc[i]  # Series

        # xtb pre-opt (neutral) -- reused as-is, no charge/multiplicity change needed
        create_sh_file(molecules_directory, i, 'xtb.sh', f'{i}.xyz', 'xtb', account=account)

        # GOAT conformer search (neutral) -- reused as-is
        create_goat_inp_file(molecules_directory, i, 'goat.inp', 'xtbopt.xyz')
        create_sh_file(molecules_directory, i, 'goat.sh', 'goat.inp', '$ORCA_PATH/orca', account=account)

        job_ID_xtb = run_sh_file(molecules_directory, i, 'xtb.sh')
        job_ID_goat = run_sh_file(molecules_directory, i, 'goat.sh', job_ID_xtb)

        # radical-cation pipeline: one FairChem opt (charge +1) + 3 DFT single points + read_files
        create_submit_cation_pipeline_sh_file(molecules_directory, i, 'submit_cation_pipeline.sh',
                                               clusternum=row['clusternum'], original_smiles=row['smiles'],
                                               batch_number=f'{batch_number:03d}', account=account)

        job_ID_pipeline = run_sh_file(molecules_directory, i, 'submit_cation_pipeline.sh', job_ID_goat)

        done_running.append(job_ID_pipeline)

    done_running = [jid for jid in done_running if jid is not None]

    batch_number += 1

    if batch_number < max_batch and done_running:
        sh_name = f'submit_batches_cations_{chain_id}.sh'
        create_submit_batches_cations_sh_file(root_directory, sh_name, batch_number,
                                               account=account, chain_id=chain_id, max_batch=max_batch)
        run_sh_file(root_directory, '', sh_name, dependency_job_id=done_running)
    else:
        print("Not submitting next batch because no valid dependency job IDs were collected.")


if __name__ == '__main__':
    main()
