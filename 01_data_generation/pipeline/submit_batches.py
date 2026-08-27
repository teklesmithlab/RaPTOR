from functions import convert_smiles_to_radicals_and_save, convert_SMILES_to_xyzs
from functions import create_sh_file, create_DFT_inp_file, run_sh_file, create_goat_inp_file
import pandas as pd
import argparse

def parse_args():

    p = argparse.ArgumentParser(
        description="split radical files after goat optimization",
    )

    p.add_argument("--root_directory", required=True, help="root file directory")
    p.add_argument("--batch_number", required=True, help="batch to run")
    p.add_argument("--account", default="tekle_smith", help="SLURM account to charge")
    p.add_argument("--chain_id", default="A", help="unique chain identifier (A or B)")
    p.add_argument("--max_batch", type=int, default=3000, help="stop chain after this batch number")
    p.add_argument("--partition", default="short", help="SLURM partition (short caps at 12h, 89 nodes)")

    return p.parse_args()


def main():

    args = parse_args()

    root_directory = args.root_directory
    batch_number = args.batch_number
    account = args.account
    chain_id = args.chain_id
    max_batch = args.max_batch
    partition = args.partition

    print(batch_number)

    batch_directory = f"{root_directory}/batches"
    molecules_directory = f"{root_directory}/molecules"

    batch_number = int(batch_number)

    smiles_file_name = f'batch_{batch_number:03d}.csv'
    print(smiles_file_name)

    # convert radical SMILES to .xyz files on cluster
    end_radical_number = convert_SMILES_to_xyzs(root_directory, smiles_file_name, 'p5_')
    print(end_radical_number)

    smiles_dataframe = pd.read_csv(f"{batch_directory}/batch_{batch_number:03d}.csv")

    done_running = []

    for i in smiles_dataframe['name']:

        smiles_dataframe1 = smiles_dataframe.set_index("name")
        row = smiles_dataframe1.loc[i]  # Series

        # create xtb sh file
        create_sh_file(molecules_directory, i, 'xtb.sh', f'{i}.xyz', 'xtb', account=account, partition=partition)

        # create goat inp and sh file
        create_goat_inp_file(molecules_directory, i, 'goat.inp', 'xtbopt.xyz')
        create_sh_file(molecules_directory, i, 'goat.sh', 'goat.inp', '$ORCA_PATH/orca', account=account, partition=partition)

        # submit xtb and goat with xtb dependency
        job_ID_xtb = run_sh_file(molecules_directory, i, 'xtb.sh')
        job_ID_goat = run_sh_file(molecules_directory, i, f'goat.sh', job_ID_xtb)

        # create sh file for splitting radicals
        create_sh_file(molecules_directory, i, f'split_goat.sh', '', 'splitgoat', goat_directory=f'{molecules_directory}/{i}/', molname=i, clusternum=row['clusternum'], original_smiles=row['smiles'], batch_number=f'{batch_number:03d}', account=account, partition=partition)

        # run python file for splitting radicals
        job_ID_splitgoat = run_sh_file(molecules_directory, i, f'split_goat.sh', job_ID_goat)

        # create sh file for running philicity calculations
        create_sh_file(molecules_directory, i, 'submit_philicities.sh', '', 'submit_philicities', dataframe_directory=f"/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/batches/batch_{batch_number:03d}_radical_dataframe.csv", batch_number=f'{batch_number:03d}', account=account, partition=partition)

        # run sh file for running and reading philicity calculations
        job_ID_philicity = run_sh_file(molecules_directory, i, 'submit_philicities.sh', job_ID_splitgoat) 

        done_running.append(job_ID_philicity)

    print(done_running)

    done_running = [jid for jid in done_running if jid is not None]

    print(done_running)

    print(batch_number+1)

    batch_number += 1
    batch_number = int(batch_number)

    print(batch_number)

    if batch_number < max_batch and done_running:
        sh_name = f'submit_batches_{chain_id}.sh'
        create_sh_file(root_directory, '', sh_name, '', 'submit_batches', batch_number=batch_number,
                       account=account, chain_id=chain_id, max_batch=max_batch, partition=partition)
        run_sh_file(root_directory, '', sh_name, dependency_job_id=done_running)

    else:
        print("Not submitting next batch because no valid dependency job IDs were collected.")


if __name__ == '__main__':

    main()

    # generate_data('/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/', '036')
