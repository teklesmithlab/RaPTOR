from __future__ import annotations

from typing import Dict, Optional, Tuple
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import random
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

RDLogger.DisableLog("rdApp.*") # disable RDKit warnings

def assign_unassigned_tetra_stereo(smiles: str, seed: Optional[int] = None, mode: str = "random") -> Optional[str]:
    """
    Assign any unassigned tetrahedral stereocenters by randomly picking one stereoisomer.
    Returns canonical SMILES with all stereocenters assigned, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    opts = StereoEnumerationOptions(unique=True, onlyUnassigned=True)
    isomers = list(EnumerateStereoisomers(mol, options=opts))

    if not isomers:
        return Chem.MolToSmiles(mol)

    rng = random.Random(seed)
    chosen = rng.choice(isomers) if mode == "random" else isomers[0]
    return Chem.MolToSmiles(chosen)


def canonicalize_smiles(smiles: str):
    """
    convert smiles to canonical smiles
    :param smiles: smiles string
    :return: returns canonical smiles string, or None if invalid
    """

    if not isinstance(smiles, str):
        print('not a valid smiles string')
        return None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True) # create rdkit mol object from smiles

        if mol is None:
            return None

        return Chem.MolToSmiles(mol, canonical=True)

    except Exception:
        print('exception occurred during canonicalization')
        return None

def canonicalize_smiles_df(df, smiles_column_name, canonical_column_name: str = "canonical_smiles", drop_bad_smiles: bool = True):
    """
    canonicalize a dataframe column of smiles strings
    :param df: pandas dataframe
    :param smiles_column_name: column name to look for smiles strings
    :param canonical_column_name: name of new column to store canonical smiles
    :param drop_bad_smiles: remove the rows with invalid smiles
    :return: original dataframe with new column of canonical smiles and invalid smiles removed (if specified)
    """

    df2 = df.copy()

    df2[canonical_column_name] = df2[smiles_column_name].apply(canonicalize_smiles)

    # find out how many smiles were sanitized by comparing the original and canonical columns
    df2['changed'] = df2[canonical_column_name] != df2[smiles_column_name]
    n_changed = df2['changed'].sum()
    print(len(df2) - n_changed, 'smiles were already canonical,', n_changed, 'smiles were changed during canonicalization')

    if drop_bad_smiles:

        n_before = len(df2)
        df2 = df2.dropna(subset=[canonical_column_name]).reset_index(drop=True)
        n_after = len(df2)

        print(f"dropped {n_before - n_after} invalid smiles")

    return df2

def fix_unassigned_stereo_rows(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    *,
    out_col: str = "SMILES_fixed",
    drop_invalid_smiles: bool = True,
    seed: Optional[int] = 0,
    mode: str = "random",
    print_stats: bool = True,
) -> pd.DataFrame:
    df2 = df.copy()

    fixed = []
    invalid_idx = []

    for idx, smi in df2[smiles_col].items():
        s_new = assign_unassigned_tetra_stereo(smi, seed=(seed + idx) if seed is not None else None, mode=mode)
        if s_new is None:
            invalid_idx.append(idx)
        fixed.append(s_new)

    df2[out_col] = fixed

    if drop_invalid_smiles:
        n_before = len(df2)
        df2 = df2.dropna(subset=[out_col]).reset_index(drop=True)
        if print_stats:
            print(f"Dropped {n_before - len(df2)} invalid SMILES")

    if print_stats:
        # how many were changed
        changed = (df2[out_col] != df2[smiles_col]).sum()
        print(f"Changed stereo/canonicalization for {changed} rows (kept the rows)")

    return df2
