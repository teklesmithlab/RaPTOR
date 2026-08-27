from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

RDLogger.DisableLog("rdApp.*")

AUGMENTATION_DIR = Path(__file__).parent.parent / "data_analysis" / "all_radicals" / "augmentation"
BATCHES_DIR = Path(__file__).parent / "batches"
COMBINED_OUT = Path(__file__).parent / "augmentation_combined.csv"

ALLOWED_ATOMS = {"C", "H", "O", "N", "F", "Cl", "Br", "I", "B", "Si", "P", "S", "Se"}
MW_DEFAULT = 200
MW_SMALL = 300
MW_SMALL_CUTOFF = 500
BATCH_SIZE = 100
NAME_PREFIX = "p5"
CLUSTERNUM_RANGE = (0, 3000)


def replace_hydrogen_isotopes(smiles: str) -> str | None:
    """Replace deuterium ([2H]) and tritium ([3H]) atoms with normal hydrogen."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    changed = False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() in (2, 3):
            atom.SetIsotope(0)
            changed = True
    if not changed:
        return smiles
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def has_abstractable_h(mol) -> bool:
    """True if the molecule has a C-H, N-H, O-H, or S-H bond (i.e. a radical can be formed)."""
    mol_h = Chem.AddHs(mol)
    for bond in mol_h.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        for x, y in ((a1, a2), (a2, a1)):
            if y.GetSymbol() == "H" and x.GetSymbol() in ("C", "N", "O", "S"):
                return True
    return False


def rdkit_ok(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if not all(atom.GetSymbol() in ALLOWED_ATOMS for atom in mol.GetAtoms()):
        return False
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > 5:
        return False
    if not has_abstractable_h(mol):
        return False
    return True


def assign_stereo(smiles: str, seed: int) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        opts = StereoEnumerationOptions(unique=True, onlyUnassigned=True)
        isomers = list(EnumerateStereoisomers(mol, options=opts))
    except RuntimeError:
        # RDKit bug on certain stereo bond configurations — return canonical SMILES as-is
        return Chem.MolToSmiles(mol)
    if not isomers:
        return Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(random.Random(seed).choice(isomers))


def filter_df(df: pd.DataFrame, mw_threshold: int) -> pd.DataFrame:
    df = df.copy()
    df["Molecular_Weight"] = pd.to_numeric(df["Molecular_Weight"], errors="coerce")
    df = df[df["Molecular_Weight"] <= mw_threshold]
    df = df[pd.to_numeric(df["Covalent_Unit_Count"], errors="coerce") <= 1]
    df = df[pd.to_numeric(df["Charge"], errors="coerce") == 0]
    df["SMILES"] = df["SMILES"].apply(lambda s: (replace_hydrogen_isotopes(s) or s) if pd.notna(s) else s)
    df = df[df["SMILES"].apply(rdkit_ok)]
    return df


def get_mw_threshold(df: pd.DataFrame) -> int:
    mw = pd.to_numeric(df["Molecular_Weight"], errors="coerce")
    return MW_SMALL if (mw <= MW_DEFAULT).sum() < MW_SMALL_CUTOFF else MW_DEFAULT


def collect_filtered_smiles() -> list[str]:
    all_smiles: list[str] = []
    for path in sorted(AUGMENTATION_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        thresh = get_mw_threshold(df)
        filtered = filter_df(df, thresh)
        smiles = filtered["SMILES"].dropna().tolist()
        print(f"  {path.name}: {len(df)} -> {len(smiles)} (MW<={thresh})")
        all_smiles.extend(smiles)
    return all_smiles


def assign_stereo_to_list(smiles_list: list[str], base_seed: int = 42) -> list[str]:
    result = []
    dropped = 0
    for i, smi in enumerate(smiles_list):
        fixed = assign_stereo(smi, seed=base_seed + i)
        if fixed is None:
            dropped += 1
        else:
            result.append(fixed)
    print(f"  Stereo assignment: {len(result)} kept, {dropped} dropped (invalid SMILES)")
    return result


def get_next_name_index() -> int:
    max_idx = -1
    pattern = re.compile(r"^batch_(\d+)$")
    for path in BATCHES_DIR.glob("batch_*.csv"):
        if not pattern.match(path.stem):
            continue
        try:
            df = pd.read_csv(path, usecols=["name"])
            for name in df["name"]:
                try:
                    idx = int(str(name).split("_")[1])
                    if idx > max_idx:
                        max_idx = idx
                except (IndexError, ValueError):
                    pass
        except Exception:
            pass
    return max_idx + 1


def get_next_batch_number() -> int:
    pattern = re.compile(r"^batch_(\d+)$")
    nums = [int(m.group(1)) for p in BATCHES_DIR.glob("batch_*.csv") if (m := pattern.match(p.stem))]
    return max(nums) + 1 if nums else 0


def main() -> None:
    print("Collecting and filtering augmentation SMILES...")
    smiles_list = collect_filtered_smiles()
    print(f"\nTotal after filtering: {len(smiles_list)}")

    print("Assigning stereochemistry...")
    smiles_list = assign_stereo_to_list(smiles_list)
    print(f"Total after stereo assignment: {len(smiles_list)}")

    combined_df = pd.DataFrame({"smiles": smiles_list})
    combined_df.to_csv(COMBINED_OUT, index=False)
    print(f"Saved combined CSV: {COMBINED_OUT} ({len(combined_df)} rows)")

    start_idx = get_next_name_index()
    next_batch = get_next_batch_number()
    print(f"Starting name: {NAME_PREFIX}_{start_idx}, starting batch: batch_{next_batch:03d}.csv\n")

    rows = [
        {
            "name": f"{NAME_PREFIX}_{start_idx + i}",
            "clusternum": random.randint(*CLUSTERNUM_RANGE),
            "smiles": smi,
        }
        for i, smi in enumerate(smiles_list)
    ]

    for j in range(0, len(rows), BATCH_SIZE):
        batch_path = BATCHES_DIR / f"batch_{next_batch:03d}.csv"
        chunk = rows[j : j + BATCH_SIZE]
        pd.DataFrame(chunk).to_csv(batch_path, index=False)
        print(f"  Created {batch_path.name} ({len(chunk)} rows)")
        next_batch += 1

    print(f"\nDone. {len(rows)} molecules written.")


if __name__ == "__main__":
    main()
