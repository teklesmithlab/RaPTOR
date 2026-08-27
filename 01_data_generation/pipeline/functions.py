from __future__ import annotations

import argparse
import csv
import fcntl
import math
import os
import re
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def append_rows_with_auto_radical_numbers_locked(
    csv_path: str,
    rows: List[Dict],
    radical_number_field: str = "radical_number",
) -> Tuple[int, int]:

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Determine fieldnames from union of keys (stable order if you want—here: first row)
    if not rows:
        return (-1, -1)

    # If you want strict columns, define them explicitly here.
    fieldnames = list(rows[0].keys())
    if radical_number_field not in fieldnames:
        fieldnames.append(radical_number_field)

    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # Read existing to find max radical_number
            f.seek(0)
            reader = csv.DictReader(f)
            max_n = -1
            if reader.fieldnames and radical_number_field in reader.fieldnames:
                for r in reader:
                    try:
                        max_n = max(max_n, int(r.get(radical_number_field, -1)))
                    except Exception:
                        pass

            start = max_n + 1

            # Ensure header exists (if file is empty)
            f.seek(0, os.SEEK_END)
            is_empty = (f.tell() == 0)
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if is_empty:
                writer.writeheader()

            # Assign and write
            n = start
            for r in rows:
                r = dict(r)  # copy so caller isn't mutated
                r[radical_number_field] = n
                writer.writerow(r)
                n += 1

            end = n - 1
            f.flush()
            os.fsync(f.fileno())
            return (start, end)

        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def get_next_radical_number_locked(csv_path: str) -> int:

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Open in a+ so file exists, and we can read then append later.
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            reader = csv.DictReader(f)
            max_n = -1
            if reader.fieldnames and "radical_number" in reader.fieldnames:
                for row in reader:
                    val = row.get("radical_number", "")
                    try:
                        max_n = max(max_n, int(val))
                    except Exception:
                        pass
            return max_n + 1
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def xyz_to_smiles(xyz_path):

    print(xyz_path)

    result = subprocess.run(
        ["obabel", xyz_path, "-osmi"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(result)

    # Open Babel prints: "SMILES<TAB>filename"
    smiles = result.stdout.strip().split()[0]
    return smiles

def create_goat_inp_file(root_directory, secondary_directory, run_file, xyz_file):

    script_content = f"""! XTB GOAT

%maxcore 2000

* xyzfile 0 1 {xyz_file}
"""

    # Build full path
    run_dir = Path(root_directory) / secondary_directory
    run_dir.mkdir(parents=True, exist_ok=True)

    run_path = run_dir / run_file

    # Write file
    with open(run_path, "w") as f:
        f.write(script_content)

    return run_path


def create_DFT_inp_file(root_directory, secondary_directory, run_file, xyz_file, charge, multiplicity):

    script_content = f"""! wB97X-3c SP
    * xyzfile {charge} {multiplicity} {xyz_file}

    %SCF 
    maxiter 1000
    end

    %maxcore 2000

    """

    # Define the path for the script (now local on cluster)
    run_file_path = Path(root_directory) / secondary_directory / run_file

    # Ensure directory exists
    run_file_path.parent.mkdir(parents=True, exist_ok=True)

    print("path:", run_file_path)

    # Write the script locally
    with open(run_file_path, "w") as f:
        f.write(script_content)

    # Make the script executable (chmod +x)
    os.chmod(run_file_path, 0o755)

def run_sh_file(root_directory, secondary_directory, sh_file, dependency_job_id=None):

    workdir = Path(root_directory) / secondary_directory
    sh_path = workdir / sh_file

    print(f"Running {sh_file} in {workdir}...")

    cmd = ["sbatch"]

    if dependency_job_id:
        if isinstance(dependency_job_id, (list, tuple, set)):
            dep_ids = ":".join(str(j) for j in dependency_job_id)
        else:
            dep_ids = str(dependency_job_id)
            
        if sh_file == 'submit_batches.sh':
            cmd.append(f"--dependency=afterany:{dep_ids}")
        else:
            cmd.append(f"--dependency=afterany:{dep_ids}")

    cmd.append(sh_path.name)

    # Execute the command
    result = subprocess.run(
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,   # we handle failure below
    )

    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()

    if output:
        print("Command Output:\n", output)
    if error:
        print("Error Output:\n", error)

    # Slurm typically prints: "Submitted batch job 123456"
    m = re.search(r"Submitted batch job\s+(\d+)", output)
    if m and result.returncode == 0:
        job_id = m.group(1)
        print(f"Job submitted with ID: {job_id}")
        return job_id

    print("Failed to submit job.")
    return None

def create_sh_file(root_directory, secondary_directory, sh_file, run_file, executable,
                   radname=None, clusternum=None, smiles=None, original_smiles=None, goat_directory=None, molname=None,
                   dataframe_directory=None, hydrogen_indices=None, specific_hydrogen=None, radical_number=None, batch_number=None,
                   account='tekle_smith', chain_id='A', max_batch=3000, partition='short'):

    root_directory = str(root_directory).rstrip("/")
    secondary_directory = str(secondary_directory).strip("/")
    experiment = sh_file.replace(".sh", "")

    workdir = Path(root_directory) / secondary_directory
    script_path = workdir / sh_file

    xtb_flags = "--gfn 2 --opt tight --charge 0" if executable == "xtb" else ""

    # Default command (non-python executables like ORCA)
    job_path =  f'{executable}  {root_directory}/{secondary_directory}/{run_file} {xtb_flags} > {experiment}.out'

    # FairChem command (your working approach)
    fairchem_job_path = (
        "python /insomnia001/depts/tekle_smith/users/MKL/project_5/models/OMol25/cli.py "
        f'--xyz "{workdir / run_file}" --charge 0 --spinmult 2 '
        f' > "{workdir /f"fairchem_{run_file}.out"}"'
    )

    # Readfiles command
    readfiles_job_path = None
    if executable == "readfiles":
        if radname is None or clusternum is None or smiles is None:
            raise ValueError("readfiles requires radname, clusternum, and smiles")
        readfiles_job_path = (
            "python /insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/read_files.py "
            f'--radname "{radname}" --clusternum "{clusternum}" --smiles "{smiles}" --original_smiles "{original_smiles}" --molname "{molname}" --hydrogen_indices "{hydrogen_indices}" --specific_hydrogen "{specific_hydrogen}" --radical_number "{radical_number}" --batch_number "{batch_number}"'
            f' > "{workdir / f"read_files_{specific_hydrogen}H.out"}"'
        )

    # splitgoat command
    splitgoat_job_path = None
    if executable == "splitgoat":
        if goat_directory is None or molname is None or clusternum is None or original_smiles is None:
            raise ValueError("splitgoat requires goat_directory, molname, clusternum, and original_smiles")
        splitgoat_job_path = (
            "python /insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/split_radicals.py "
            f'--goat_directory "{workdir}" --molname "{secondary_directory}" --clusternum "{clusternum}" --original_smiles "{original_smiles}" --batch_number "{batch_number}"'
            f' > "{workdir / "splitgoat.out"}"'
        )
    
    # submit philicities command
    submit_philicities_job_path = None
    if executable == "submit_philicities":
        if dataframe_directory is None:
            raise ValueError("submit philicities requires a dataframe directory")
        submit_philicities_job_path = (
            "python /insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/submit_philicities.py "
            f'--dataframe_directory "{dataframe_directory}" --root_directory "{root_directory}" --secondary_directory "{secondary_directory}" --batch_number "{batch_number}" --account "{account}" --partition "{partition}"'
            f' > "{workdir / "submit_philicities.out"}"'
        )
    
    # delete files command
    delete_files_job_path = None
    if executable == "delete_files":
        if molname is None:
            raise ValueError("delete files requires a dataframe directory")
        delete_files_job_path = (
            "python /insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/delete_files.py "
            f'--molname "{molname}"'
            f' > "{workdir / "delete_files.out"}"'
        )

    # submit batches command
    submit_batches_job_path = None
    if executable == "submit_batches":
        if batch_number is None:
            raise ValueError("submit batches requires a batch number")
        submit_batches_job_path = (
            "python submit_batches.py "
            f'--root_directory "{root_directory}" --batch_number {batch_number} '
            f'--account "{account}" --chain_id "{chain_id}" --max_batch {max_batch} '
            f'--partition "{partition}"'
            f' > "{workdir / f"submit_batches_{chain_id}.out"}"'
        )

    # Pick exactly ONE command
    if executable == "fairchem":
        run_cmd = fairchem_job_path
    elif executable == "readfiles":
        run_cmd = readfiles_job_path
    elif executable == 'splitgoat':
        run_cmd = splitgoat_job_path
    elif executable == 'submit_philicities':
        run_cmd = submit_philicities_job_path
    elif executable == 'delete_files':
        run_cmd = delete_files_job_path
    elif executable == 'submit_batches':
        run_cmd = submit_batches_job_path
    else:
        run_cmd = job_path

    # Only source/conda when needed
    bashrc_line = "source ~/.bashrc" if executable in ("fairchem", "readfiles", "splitgoat", "submit_philicities", "delete_files", "submit_batches") else ""

    if executable == "splitgoat":
        conda_line = "conda activate chem_obabel"
        submit_batches1 = ""
        submit_batches2 = ""
    elif executable == "fairchem":
        conda_line = "conda activate fairchem"
        submit_batches1 = ""
        submit_batches2 = ""
    elif executable == "submit_batches":
        conda_line = "conda activate ~/envs/chem"
        submit_batches1 = "module purge"
        submit_batches2 = "module load anaconda/2023.09"

    else:
        conda_line = ""
        submit_batches1 = ""
        submit_batches2 = ""

    script_content = f"""#!/bin/bash

#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --job-name={secondary_directory}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-2:00

export NBOEXE=/insomnia001/depts/tekle_smith/users/softwares/NBO/nbo7/bin/nbo7.i8.exe
export NBO7KEY=/insomnia001/depts/tekle_smith/users/softwares/NBO/nbo7/nbo7.key

{submit_batches1}
{submit_batches2}
{bashrc_line}
{conda_line}

{run_cmd}
"""

    # Write on cluster/local FS (no SFTP)
    workdir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)

    # Make executable
    script_path.chmod(0o755)

    return str(script_path)


def save_mol_to_xyz(RDKit_molecule, xyz_filepath) -> bool:
    """
    Returns True if XYZ successfully written, otherwise False.
    Catches RDKit conformer/optimization failures (e.g., ValueError: Bad Conformer Id)
    and continues gracefully.
    """
    if RDKit_molecule is None:
        return False

    try:
        mol = Chem.AddHs(RDKit_molecule)

        # Embed; if it fails, status != 0
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status != 0:
            return False

        # Optimize; MMFF sometimes throws or returns non-zero
        try:
            mmff_status = AllChem.MMFFOptimizeMolecule(mol)
            # mmff_status: 0 converged, 1 not converged, -1 error
            if mmff_status == -1:
                return False
        except Exception:
            # this is where "ValueError: Bad Conformer Id" will be caught
            return False

        conf = mol.GetConformer()  # can also throw if no conformer
    except Exception:
        return False

    # Only write if we successfully got coordinates
    with open(xyz_filepath, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n")

    return True


def convert_SMILES_to_xyzs(root_directory, smiles_file_name, root_folder_name):
    smiles_csv_path = Path(f'{root_directory}/batches/{smiles_file_name}')
    xyz_file_path = Path(f'{root_directory}/molecules/')

    df = pd.read_csv(smiles_csv_path)

    # assumes columns: 'smiles', 'name'
    smiles_strings = df["smiles"].astype(str)
    molecule_names = df["name"].astype(str)

    for name, smiles in zip(molecule_names, smiles_strings):
        smiles = smiles.strip()
        if not smiles:
            continue

        RDKit_molecule = Chem.MolFromSmiles(smiles)
        if RDKit_molecule is None:
            print(f"Failed to parse SMILES for {name}: {smiles}")
            continue

        xyz_dir = xyz_file_path / name
        xyz_dir.mkdir(parents=True, exist_ok=True)

        xyz_path = xyz_dir / f"{name}.xyz"

        ok = save_mol_to_xyz(RDKit_molecule, str(xyz_path))
        if ok:
            print(f"Wrote {xyz_path}")
        else:
            # optional: delete an empty/partial xyz if it was created (we don't create it on failure below)
            print(f"Skipping {name} due to 3D/optimization failure.")

    print("Done!")
    return None

def get_carbon_hydrogen_map(smiles):

    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)  # add explicit hydrogens

    carbon_h_map = {}

    for atom in m.GetAtoms():
        if atom.GetSymbol() == 'C':
            carbon_idx = atom.GetIdx() + 1  # +1 for human-readable indexing
            hydrogens = [
                neighbor.GetIdx() + 1
                for bond in atom.GetBonds()
                for neighbor in [bond.GetOtherAtom(atom)]
                if neighbor.GetSymbol() == 'H'
            ]
            carbon_h_map[carbon_idx] = hydrogens

    return carbon_h_map

def generate_radical_smiles(smiles, index_dict):

    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    radical_smiles_list = []

    for carbon_idx, hydrogens in index_dict.items():
        if len(hydrogens) > 0:
            mol_copy = Chem.RWMol(m)
            atom = mol_copy.GetAtomWithIdx(carbon_idx - 1)
            atom.SetAtomicNum(6)  # still carbon

            # remove one hydrogen
            neighbors = [n for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
            if neighbors:
                mol_copy.RemoveAtom(neighbors[0].GetIdx())

            # mark as radical
            atom.SetNumRadicalElectrons(1)
            Chem.SanitizeMol(mol_copy)

            mol_copy = Chem.RemoveHs(mol_copy)

            try:
                Chem.SanitizeMol(mol_copy)
                radical_smiles = Chem.MolToSmiles(mol_copy)
                radical_smiles_list.append(radical_smiles)
            except:
                continue

    # remove duplicates
    radical_smiles_list = list(set(radical_smiles_list))
    return radical_smiles_list

def convert_smiles_to_radicals_and_save(smile_file_directory, output_file_directory, radical_number):

    df = pd.read_csv(smile_file_directory)

    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain a column named 'smiles'")

    results = []

    for clusternum, smiles in df[["clusternum", "smiles"]].itertuples(index=False):

        if pd.isna(smiles):
            continue

        try:
            carbon_h_map = get_carbon_hydrogen_map(smiles)
            radical_list = generate_radical_smiles(smiles, carbon_h_map)

            for radical in radical_list:
                results.append({
                    "radname": f"p5_{radical_number}",
                    "clusternum": clusternum,
                    "smiles": radical,
                    "original_smiles": smiles
                })
                radical_number += 1   # 🔑 increment per row

        except Exception as e:
            print(f"skipping {smiles} due to error: {e}")

    radical_df = pd.DataFrame(results)
    radical_df = radical_df.drop_duplicates().reset_index(drop=True)
    radical_df.to_csv(output_file_directory, index=False)

    print(f"radical CSV saved: {output_file_directory}")

    return radical_df

def to_canonical_smiles(smiles: str) -> str:

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # invalid SMILES
    return Chem.MolToSmiles(mol, canonical=True)

def check_normal_termination(filepath) -> bool:
    """Return True if the ORCA output file contains 'ORCA TERMINATED NORMALLY'."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return any('ORCA TERMINATED NORMALLY' in line for line in lines[-20:])
    except Exception:
        return False


def read_spin_contamination(filepath) -> float | None:
    """Return the last <S**2> value from an ORCA radical output, or None if not found."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if 'Expectation value of <S**2>' in line:
                return float(line.split()[-1])
    except Exception:
        return None
    return None


def read_scf_energy(file_directory):

    with open(file_directory, 'r') as file:
        lines = file.readlines()

    for line in reversed(lines):  # Start from the bottom
        if 'FINAL SINGLE POINT ENERGY' in line:
            return float(line.split()[-1])

    print("SCF energy not found.")
    return None

def read_philicity(folder_directory, specific_hydrogen):

    anion_scf = read_scf_energy(f'{folder_directory}/DFT_elec_anion_{specific_hydrogen}H.out')
    cation_scf = read_scf_energy(f'{folder_directory}/DFT_elec_cation_{specific_hydrogen}H.out')
    radical_scf = read_scf_energy(f'{folder_directory}/DFT_elec_radical_{specific_hydrogen}H.out')

    I = (cation_scf - radical_scf)*27.2114
    A = (radical_scf - anion_scf)*27.2114

    philicity = ((I + A)**2) / (I - A) * (1 / 8)  # convert to eV

    return I, A, philicity

def split_radicals(goat_file_directory):

    goat_dir = Path(goat_file_directory)
    xyz_file_path = goat_dir / "goat.globalminimum.xyz"

    if not xyz_file_path.exists():
        print(f"Failed to find {xyz_file_path}, skipping.")
        return []

    lines = xyz_file_path.read_text().splitlines(keepends=True)
    if len(lines) < 3:
        print(f"{xyz_file_path} doesn't look like a valid XYZ file (too few lines).")
        return []

    # Parse header
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        print(f"First line of {xyz_file_path} is not an integer atom count.")
        return []

    comment_line = lines[1]
    atom_lines = lines[2:]

    if len(atom_lines) != num_atoms:
        print(
            f"Warning: header says {num_atoms} atoms but file has {len(atom_lines)} atom lines. "
            f"Proceeding with atom_lines length."
        )
        num_atoms = len(atom_lines)

    # Identify hydrogen indices (0-based into atom_lines)
    hydrogen_indices = [
        i for i, line in enumerate(atom_lines)
        if line.strip().startswith("H")
    ]

    written_labels = []

    for h_index in hydrogen_indices:
        new_atom_lines = atom_lines[:h_index] + atom_lines[h_index + 1:]
        new_num_atoms_line = f"{num_atoms - 1}\n"
        new_lines = [new_num_atoms_line, comment_line] + new_atom_lines

        label = h_index + 1          # 👈 this matches the filename
        output_file = goat_dir / f"{label}H.xyz"
        output_file.write_text("".join(new_lines))

        written_labels.append(label)

    print(f"Generated {len(written_labels)} radicals in {goat_dir}")
    return written_labels


def append_row_locked(csv_path: Path, row: dict, lock_path: Path | None = None):

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a dedicated lock file (recommended)
    lock_path = Path(lock_path) if lock_path else csv_path.with_suffix(csv_path.suffix + ".lock")

    # Open lock file and lock it
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        file_exists = csv_path.exists()

        # Open CSV and append
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))

            # Only write header if file is new/empty
            if (not file_exists) or csv_path.stat().st_size == 0:
                writer.writeheader()

            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

        # Unlock
        fcntl.flock(lock_f, fcntl.LOCK_UN)


def split_radicals_sulfur_h_only(goat_file_directory, sh_max_dist=1.5, heavy_max_dist=1.8):

    goat_dir = Path(goat_file_directory)
    xyz_file_path = goat_dir / "goat.globalminimum.xyz"

    if not xyz_file_path.exists():
        print(f"Failed to find {xyz_file_path}, skipping.")
        return []

    lines = xyz_file_path.read_text().splitlines(keepends=True)
    if len(lines) < 3:
        print(f"{xyz_file_path} doesn't look like a valid XYZ file (too few lines).")
        return []

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        print(f"First line of {xyz_file_path} is not an integer atom count.")
        return []

    comment_line = lines[1]
    atom_lines = lines[2:]

    if len(atom_lines) != num_atoms:
        print(
            f"Warning: header says {num_atoms} atoms but file has {len(atom_lines)} atom lines. "
            f"Proceeding with atom_lines length."
        )
        num_atoms = len(atom_lines)

    # Parse atoms
    elems = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            # skip malformed
            elems.append(None)
            coords.append([np.nan, np.nan, np.nan])
            continue
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords, dtype=float)

    # Indices
    h_indices = [i for i, e in enumerate(elems) if e == "H"]
    heavy_indices = [i for i, e in enumerate(elems) if e != "H" and e is not None]

    sulfur_h_atomline_indices = []

    for hi in h_indices:
        h_xyz = coords[hi]
        if not np.isfinite(h_xyz).all():
            continue

        # find nearest heavy atom
        best_j = None
        best_d = 1e9
        for j in heavy_indices:
            d = float(np.linalg.norm(coords[j] - h_xyz))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is None:
            continue

        # Only accept if nearest heavy atom is carbon and within a C–H bond distance
        if elems[best_j] == "S" and best_d <= sh_max_dist:
            sulfur_h_atomline_indices.append(hi)

    written_labels = []

    for h_index in sulfur_h_atomline_indices:
        new_atom_lines = atom_lines[:h_index] + atom_lines[h_index + 1:]
        new_num_atoms_line = f"{num_atoms - 1}\n"
        new_lines = [new_num_atoms_line, comment_line] + new_atom_lines

        label = h_index + 1  # matches your filename convention (line index + 1)
        output_file = goat_dir / f"{label}H.xyz"
        output_file.write_text("".join(new_lines))
        written_labels.append(label)

    print(f"Generated {len(written_labels)} *S–H-only* radicals in {goat_dir}")
    return written_labels

def split_radicals_oxygen_h_only(goat_file_directory, oh_max_dist=1.05, heavy_max_dist=1.8):

    goat_dir = Path(goat_file_directory)
    xyz_file_path = goat_dir / "goat.globalminimum.xyz"

    if not xyz_file_path.exists():
        print(f"Failed to find {xyz_file_path}, skipping.")
        return []

    lines = xyz_file_path.read_text().splitlines(keepends=True)
    if len(lines) < 3:
        print(f"{xyz_file_path} doesn't look like a valid XYZ file (too few lines).")
        return []

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        print(f"First line of {xyz_file_path} is not an integer atom count.")
        return []

    comment_line = lines[1]
    atom_lines = lines[2:]

    if len(atom_lines) != num_atoms:
        print(
            f"Warning: header says {num_atoms} atoms but file has {len(atom_lines)} atom lines. "
            f"Proceeding with atom_lines length."
        )
        num_atoms = len(atom_lines)

    # Parse atoms
    elems = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            # skip malformed
            elems.append(None)
            coords.append([np.nan, np.nan, np.nan])
            continue
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords, dtype=float)

    # Indices
    h_indices = [i for i, e in enumerate(elems) if e == "H"]
    heavy_indices = [i for i, e in enumerate(elems) if e != "H" and e is not None]

    oxygen_h_atomline_indices = []

    for hi in h_indices:
        h_xyz = coords[hi]
        if not np.isfinite(h_xyz).all():
            continue

        # find nearest heavy atom
        best_j = None
        best_d = 1e9
        for j in heavy_indices:
            d = float(np.linalg.norm(coords[j] - h_xyz))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is None:
            continue

        # Only accept if nearest heavy atom is carbon and within a C–H bond distance
        if elems[best_j] == "O" and best_d <= oh_max_dist:
            oxygen_h_atomline_indices.append(hi)

    written_labels = []

    for h_index in oxygen_h_atomline_indices:
        new_atom_lines = atom_lines[:h_index] + atom_lines[h_index + 1:]
        new_num_atoms_line = f"{num_atoms - 1}\n"
        new_lines = [new_num_atoms_line, comment_line] + new_atom_lines

        label = h_index + 1  # matches your filename convention (line index + 1)
        output_file = goat_dir / f"{label}H.xyz"
        output_file.write_text("".join(new_lines))
        written_labels.append(label)

    print(f"Generated {len(written_labels)} *N–H-only* radicals in {goat_dir}")
    return written_labels


def split_radicals_nitrogen_h_only(goat_file_directory, nh_max_dist=1.25, heavy_max_dist=1.8):

    goat_dir = Path(goat_file_directory)
    xyz_file_path = goat_dir / "goat.globalminimum.xyz"

    if not xyz_file_path.exists():
        print(f"Failed to find {xyz_file_path}, skipping.")
        return []

    lines = xyz_file_path.read_text().splitlines(keepends=True)
    if len(lines) < 3:
        print(f"{xyz_file_path} doesn't look like a valid XYZ file (too few lines).")
        return []

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        print(f"First line of {xyz_file_path} is not an integer atom count.")
        return []

    comment_line = lines[1]
    atom_lines = lines[2:]

    if len(atom_lines) != num_atoms:
        print(
            f"Warning: header says {num_atoms} atoms but file has {len(atom_lines)} atom lines. "
            f"Proceeding with atom_lines length."
        )
        num_atoms = len(atom_lines)

    # Parse atoms
    elems = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            # skip malformed
            elems.append(None)
            coords.append([np.nan, np.nan, np.nan])
            continue
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords, dtype=float)

    # Indices
    h_indices = [i for i, e in enumerate(elems) if e == "H"]
    heavy_indices = [i for i, e in enumerate(elems) if e != "H" and e is not None]

    nitrogen_h_atomline_indices = []

    for hi in h_indices:
        h_xyz = coords[hi]
        if not np.isfinite(h_xyz).all():
            continue

        # find nearest heavy atom
        best_j = None
        best_d = 1e9
        for j in heavy_indices:
            d = float(np.linalg.norm(coords[j] - h_xyz))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is None:
            continue

        # Only accept if nearest heavy atom is carbon and within a C–H bond distance
        if elems[best_j] == "N" and best_d <= nh_max_dist:
            nitrogen_h_atomline_indices.append(hi)

    written_labels = []

    for h_index in nitrogen_h_atomline_indices:
        new_atom_lines = atom_lines[:h_index] + atom_lines[h_index + 1:]
        new_num_atoms_line = f"{num_atoms - 1}\n"
        new_lines = [new_num_atoms_line, comment_line] + new_atom_lines

        label = h_index + 1  # matches your filename convention (line index + 1)
        output_file = goat_dir / f"{label}H.xyz"
        output_file.write_text("".join(new_lines))
        written_labels.append(label)

    print(f"Generated {len(written_labels)} *N–H-only* radicals in {goat_dir}")
    return written_labels

def split_radicals_heteroatom_h(goat_file_directory, bond_distances=None):
    """
    Split radicals for all heteroatom X-H bonds in one pass.
    bond_distances maps element symbol -> max X-H bond distance (Angstrom).
    """
    if bond_distances is None:
        bond_distances = {"N": 1.25, "O": 1.05, "S": 1.50}

    goat_dir = Path(goat_file_directory)
    xyz_file_path = goat_dir / "goat.globalminimum.xyz"

    if not xyz_file_path.exists():
        print(f"Failed to find {xyz_file_path}, skipping.")
        return []

    lines = xyz_file_path.read_text().splitlines(keepends=True)
    if len(lines) < 3:
        print(f"{xyz_file_path} doesn't look like a valid XYZ file (too few lines).")
        return []

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        print(f"First line of {xyz_file_path} is not an integer atom count.")
        return []

    comment_line = lines[1]
    atom_lines = lines[2:]

    if len(atom_lines) != num_atoms:
        print(
            f"Warning: header says {num_atoms} atoms but file has {len(atom_lines)} atom lines. "
            f"Proceeding with atom_lines length."
        )
        num_atoms = len(atom_lines)

    elems = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            elems.append(None)
            coords.append([np.nan, np.nan, np.nan])
            continue
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords, dtype=float)

    h_indices = [i for i, e in enumerate(elems) if e == "H"]
    heavy_indices = [i for i, e in enumerate(elems) if e != "H" and e is not None]

    matched_indices = []

    for hi in h_indices:
        h_xyz = coords[hi]
        if not np.isfinite(h_xyz).all():
            continue

        best_j = None
        best_d = 1e9
        for j in heavy_indices:
            d = float(np.linalg.norm(coords[j] - h_xyz))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is None:
            continue

        elem = elems[best_j]
        if elem in bond_distances and best_d <= bond_distances[elem]:
            matched_indices.append(hi)

    written_labels = []

    for h_index in matched_indices:
        new_atom_lines = atom_lines[:h_index] + atom_lines[h_index + 1:]
        new_num_atoms_line = f"{num_atoms - 1}\n"
        new_lines = [new_num_atoms_line, comment_line] + new_atom_lines

        label = h_index + 1
        output_file = goat_dir / f"{label}H.xyz"
        output_file.write_text("".join(new_lines))
        written_labels.append(label)

    print(f"Generated {len(written_labels)} heteroatom X-H radicals {list(bond_distances.keys())} in {goat_dir}")
    return written_labels


def split_radicals_carbon_h_only(goat_file_directory, ch_max_dist=1.25, heavy_max_dist=1.8):

    goat_dir = Path(goat_file_directory)
    xyz_file_path = goat_dir / "goat.globalminimum.xyz"

    if not xyz_file_path.exists():
        print(f"Failed to find {xyz_file_path}, skipping.")
        return []

    lines = xyz_file_path.read_text().splitlines(keepends=True)
    if len(lines) < 3:
        print(f"{xyz_file_path} doesn't look like a valid XYZ file (too few lines).")
        return []

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        print(f"First line of {xyz_file_path} is not an integer atom count.")
        return []

    comment_line = lines[1]
    atom_lines = lines[2:]

    if len(atom_lines) != num_atoms:
        print(
            f"Warning: header says {num_atoms} atoms but file has {len(atom_lines)} atom lines. "
            f"Proceeding with atom_lines length."
        )
        num_atoms = len(atom_lines)

    # Parse atoms
    elems = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            # skip malformed
            elems.append(None)
            coords.append([np.nan, np.nan, np.nan])
            continue
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords, dtype=float)

    # Indices
    h_indices = [i for i, e in enumerate(elems) if e == "H"]
    heavy_indices = [i for i, e in enumerate(elems) if e != "H" and e is not None]

    carbon_h_atomline_indices = []

    for hi in h_indices:
        h_xyz = coords[hi]
        if not np.isfinite(h_xyz).all():
            continue

        # find nearest heavy atom
        best_j = None
        best_d = 1e9
        for j in heavy_indices:
            d = float(np.linalg.norm(coords[j] - h_xyz))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is None:
            continue

        # Only accept if nearest heavy atom is carbon and within a C–H bond distance
        if elems[best_j] == "C" and best_d <= ch_max_dist:
            carbon_h_atomline_indices.append(hi)

    written_labels = []

    for h_index in carbon_h_atomline_indices:
        new_atom_lines = atom_lines[:h_index] + atom_lines[h_index + 1:]
        new_num_atoms_line = f"{num_atoms - 1}\n"
        new_lines = [new_num_atoms_line, comment_line] + new_atom_lines

        label = h_index + 1  # matches your filename convention (line index + 1)
        output_file = goat_dir / f"{label}H.xyz"
        output_file.write_text("".join(new_lines))
        written_labels.append(label)

    print(f"Generated {len(written_labels)} *C–H-only* radicals in {goat_dir}")
    return written_labels

def combine_finished_batches(highest_batch_number: int, directory: str, output_name: str = "all_philicities.csv"):

    REQUIRED_COLUMNS = [
    "original_smiles",
    "radical_name",
    "cluster_assignment",
    "radical_smiles",
    "philicity",
    "I",
    "A"]

    dfs = []

    for i in range(2415, highest_batch_number + 1):
        fname = f"batch_{i:03d}_philicities.csv"
        fpath = os.path.join('/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/batches', fname)

        if not os.path.exists(fpath):
            print(f"skipping missing file: {fname}")
            continue

        df = pd.read_csv(fpath)

        # check required columns exist
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"skipping {fname}, missing columns: {missing}")
            continue

        # extract only required columns, in the correct order
        sub = df[REQUIRED_COLUMNS].copy()
        sub["batch_id"] = i

        dfs.append(sub)

    if not dfs:
        raise RuntimeError("No valid batch files found to combine.")

    combined = pd.concat(dfs, ignore_index=True)

    # drop rows missing essential data
    combined = combined.dropna(
        subset=["original_smiles", "radical_smiles", "philicity"],
        how="any",
    ).reset_index(drop=True)

    out_path = os.path.join(directory, output_name)
    combined.to_csv(out_path, index=False)

    print(f"combined {len(dfs)} batches → {out_path}")
    print(f"   total rows: {len(combined)}")

    return combined


def find_imaginary_nodes(directory, hydrogen):
    """
    Check whether the ORCA output file contains imaginary modes.

    Parameters
    ----------
    directory : str
        Path to directory containing output file
    hydrogen : str or int
        Hydrogen identifier used in filename

    Returns
    -------
    bool
        True if imaginary modes > 0, False otherwise
    """
    filename = f"fairchem_{hydrogen}H.xyz.out"
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        for line in f:
            if "Total imaginary modes:" in line:
                # extract integer at end of line
                modes = int(re.search(r"(\d+)", line).group(1))
                return modes > 0

    # If the line was never found, assume no imaginary modes
    return False

def delete_philicity_rows_for_radicals(
    targets_df: pd.DataFrame,
    batches_dir: Union[str, Path],
    philicity_filename_template: str = "batch_{batch:03d}_philicities.csv",
    smiles_col_targets: str = "radical_smiles",
    batch_col_targets: str = "batch",
    smiles_col_results: str = "radical_smiles",
) -> None:
    """
    Delete rows from per-batch philicity CSV files based on (batch, radical_smiles),
    printing which SMILES were deleted and which were not found.
    """
    batches_dir = Path(batches_dir)

    # sanity check
    required = {smiles_col_targets, batch_col_targets}
    missing = required - set(targets_df.columns)
    if missing:
        raise KeyError(f"targets_df missing columns: {sorted(missing)}")

    # normalize
    targets_df = targets_df.copy()
    targets_df[batch_col_targets] = targets_df[batch_col_targets].astype(int)
    targets_df[smiles_col_targets] = targets_df[smiles_col_targets].astype(str)

    # group targets by batch
    for batch, group in targets_df.groupby(batch_col_targets):
        smiles_targets = list(group[smiles_col_targets])

        results_path = batches_dir / philicity_filename_template.format(batch=batch)

        if not results_path.exists():
            print(f"[batch {batch:03d}] philicity file missing → skipping")
            continue

        with open(results_path, "r+", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            f.seek(0)
            df = pd.read_csv(f)

            if smiles_col_results not in df.columns:
                print(f"[batch {batch:03d}] missing column '{smiles_col_results}' → skipping")
                fcntl.flock(f, fcntl.LOCK_UN)
                continue

            existing_smiles = set(df[smiles_col_results].astype(str))

            # track deletions
            deleted = []
            not_found = []

            for smi in smiles_targets:
                if smi in existing_smiles:
                    deleted.append(smi)
                else:
                    not_found.append(smi)

            if deleted:
                df = df[~df[smiles_col_results].astype(str).isin(deleted)]

                print(f"[batch {batch:03d}] deleted {len(deleted)} rows:")
                for smi in deleted:
                    print(f"  ✔ deleted: {smi}")

                # rewrite file
                f.seek(0)
                f.truncate(0)
                df.to_csv(f, index=False)

            if not_found:
                for smi in not_found:
                    print(f"  ⚠ not found: {smi}")

            if not deleted and not not_found:
                print(f"[batch {batch:03d}] no matching radicals")

            fcntl.flock(f, fcntl.LOCK_UN)



if __name__ == '__main__':

    df = combine_finished_batches(3000, '/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/batches')