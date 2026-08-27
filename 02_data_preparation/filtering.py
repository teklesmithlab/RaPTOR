from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Fragments
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

def if_has_rotatable_bond_count(mol, min_count=0, max_count=10):

    rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    return min_count <= rotatable_bonds <= max_count

def if_has_only_allowed_atoms(mol, allowed={'C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'P', 'S', 'Se'}):

    return all(atom.GetSymbol() in allowed for atom in mol.GetAtoms())

def if_has_nh_bond(mol):

    mol = Chem.AddHs(mol)

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()

        # Check that the hydrogen is protium (¹H), not deuterium (²H)
        if (
            (a1.GetSymbol() == 'N' and a2.GetSymbol() == 'H' and a2.GetIsotope() in [0, 1]) or
            (a2.GetSymbol() == 'N' and a1.GetSymbol() == 'H' and a1.GetIsotope() in [0, 1])
        ):
            return True

    return False

def if_has_oh_bond(mol):

    mol = Chem.AddHs(mol)

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()

        # Check that the hydrogen is protium (¹H), not deuterium (²H)
        if (
            (a1.GetSymbol() == 'O' and a2.GetSymbol() == 'H' and a2.GetIsotope() in [0, 1]) or
            (a2.GetSymbol() == 'O' and a1.GetSymbol() == 'H' and a1.GetIsotope() in [0, 1])
        ):
            return True

    return False

def filter_and_save_smiles(smile_file_directory, output_file_directory, min_rotatable=0, max_rotatable=10, max_heavy_atoms=20):

    df = pd.read_csv(smile_file_directory)

    print(f'initial dataset size: {len(df["SMILES"])}')

    if 'SMILES' not in df.columns:
        raise ValueError("CSV must contain a column named 'smiles'")

    filtered_rows = []

    for smiles in df['SMILES']:

        if pd.isna(smiles):
            continue

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        # filter 1: must contain only allowed atoms
        if not if_has_only_allowed_atoms(mol):
            continue

        # filter 2: must contain at least one N–H bond
        if not if_has_oh_bond(mol):
            continue

        # filter 3: can't have a salt present
        if '.' in smiles:
            continue

        # filter 4: can't have an overall charge
        if Chem.GetFormalCharge(mol) != 0:
            continue

        # filter 5: must have rotatable bond count within range
        if not if_has_rotatable_bond_count(mol, min_rotatable, max_rotatable):
            continue

        # filter 6: must have less than 15 heavy atoms
        if mol.GetNumHeavyAtoms() >= max_heavy_atoms:
            continue

        print(f'{smiles} passed all filters')
        filtered_rows.append(smiles)

    # ensure all smiles are unique (no duplicates)
    filtered_rows = list(set(filtered_rows))

    # save results
    result_df = pd.DataFrame(filtered_rows, columns=['smiles'])
    result_df.to_csv(output_file_directory, index=False)
    print(f"filtering complete. {len(result_df)} molecules saved to {output_file_directory}")

def count_stereocenters_from_smiles(smiles: str) -> Dict[str, int]:
    """
    Count assigned and unassigned atom- and bond-level stereocenters
    for a SMILES string using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "atom_assigned": 0,
            "atom_unassigned": 0,
            "bond_assigned": 0,
            "bond_unassigned": 0,
            "invalid": True,
        }

    # Ensure stereo perception
    Chem.FindPotentialStereoBonds(mol)

    stereocenters = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True
    )

    stereobonds = [
        bond
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE
    ]

    atom_assigned = sum(1 for _, tag in stereocenters if tag != "?")
    atom_unassigned = sum(1 for _, tag in stereocenters if tag == "?")

    bond_assigned = sum(
        1
        for bond in stereobonds
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREOANY
    )
    bond_unassigned = sum(
        1
        for bond in stereobonds
        if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOANY
    )

    return {
        "atom_assigned": atom_assigned,
        "atom_unassigned": atom_unassigned,
        "bond_assigned": bond_assigned,
        "bond_unassigned": bond_unassigned,
        "invalid": False,
    }

def has_too_many_unassigned_stereocenters(smiles: str, max_unassigned: int = 1) -> bool:
    """
    Returns True if the molecule has more than `max_unassigned`
    unassigned stereocenters (atoms + bonds).
    """
    counts = count_stereocenters_from_smiles(smiles)

    if counts["invalid"]:
        return True  # drop invalid SMILES by default

    total_unassigned = counts["atom_unassigned"] + counts["bond_unassigned"]

    if total_unassigned > max_unassigned:
        print(smiles)

    return total_unassigned > max_unassigned

def remove_molecules_with_no_stereochemistry(
    root_dir,
    *,
    smiles_col: str = "original_smiles",
    output_suffix: str = "_stereo_filtered",
    max_unassigned: int = 1,
    dry_run: bool = True,
):
    """
    Given a directory containing CSV files (or a single CSV),
    remove rows whose `smiles_col` contains more than `max_unassigned`
    unassigned stereocenters.

    If dry_run=True, prints a summary but does NOT overwrite files.
    """

    root_dir = Path(root_dir)

    csv_files = (
        [root_dir]
        if root_dir.is_file() and root_dir.suffix == ".csv"
        else sorted(root_dir.glob("*.csv"))
    )

    if not csv_files:
        raise ValueError("No CSV files found.")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        if smiles_col not in df.columns:
            print(f"Skipping {csv_path.name} (no '{smiles_col}' column)")
            continue

        mask_drop = df[smiles_col].apply(
            lambda smi: has_too_many_unassigned_stereocenters(
                smi, max_unassigned=max_unassigned
            )
        )

        n_total = len(df)
        n_drop = int(mask_drop.sum())
        n_keep = n_total - n_drop

        print(
            f"{csv_path.name}: "
            f"drop {n_drop}/{n_total} "
            f"({100 * n_drop / n_total:.1f}%)"
        )

        if dry_run:
            continue

        filtered_df = df.loc[~mask_drop].reset_index(drop=True)

        filtered_df.to_csv(root_dir, index=False)

        print(f"  → wrote {root_dir}")
