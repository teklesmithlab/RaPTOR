import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem

import sys
sys.path.insert(0, '/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation')
from functions import read_spin_contamination, read_philicity

def count_stereocenters_from_smiles(smiles: str) -> Dict[str, int]:

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Ensure RDKit has identified potential stereo bonds
    Chem.FindPotentialStereoBonds(mol)

    stereocenters = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    stereobonds = [
        bond
        for bond in mol.GetBonds()
        if bond.GetStereo() is not Chem.rdchem.BondStereo.STEREONONE
    ]

    atom_assigned = sum(1 for _, tag in stereocenters if tag != "?")
    atom_unassigned = sum(1 for _, tag in stereocenters if tag == "?")

    bond_assigned = sum(
        1
        for bond in stereobonds
        if bond.GetStereo() is not Chem.rdchem.BondStereo.STEREOANY
    )
    bond_unassigned = sum(
        1 for bond in stereobonds if bond.GetStereo() is Chem.rdchem.BondStereo.STEREOANY
    )

    return {
        "atom_assigned": atom_assigned,
        "atom_unassigned": atom_unassigned,
        "bond_assigned": bond_assigned,
        "bond_unassigned": bond_unassigned,
    }

def filter_philicity(
    df: pd.DataFrame,
    col: str = "philicity",
    low: float = 0.2,
    high: float = 4.0,
    cation_high: float = 20.0,
    smiles_col: str = "radical_smiles",
    safe: bool = True,
) -> pd.DataFrame:
    """
    Remove rows where `col` is outside [low, high].

    Amine radical cations (detected by a '+' formal charge in radical_smiles)
    use a wider upper bound (cation_high) because the shifted charge ladder
    (neutral/radical-cation/dication) produces philicity values of ~5-15 eV,
    well above the neutral-radical range of [0.2, 4.0] eV.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    is_cation = df[smiles_col].str.contains(r"\+", na=False) if smiles_col in df.columns else pd.Series(False, index=df.index)
    effective_high = is_cation.map({True: cation_high, False: high})

    mask_bad = (df[col] < low) | (df[col] > effective_high)
    n_neutral_bad = int((mask_bad & ~is_cation).sum())
    n_cation_bad = int((mask_bad & is_cation).sum())

    print(f"Deleted {mask_bad.sum()} rows on philicity bounds "
          f"(neutral [{low}, {high}]: {n_neutral_bad} removed; "
          f"cation [{low}, {cation_high}]: {n_cation_bad} removed)")

    if safe:
        return df.loc[~mask_bad].copy()
    else:
        df.drop(index=df.index[mask_bad], inplace=True)
        return df

def filter_unassigned_stereo_rows(
    df: pd.DataFrame,
    smiles_col: str = "original_smiles",
    *,
    drop_invalid_smiles: bool = True,
    keep_stereo_counts: bool = False,
    print_dropped: bool = True,
    max_print: int = 50,
) -> pd.DataFrame:
    """
    Drop rows where atom or bond stereochemistry is unassigned.
    Optionally drops invalid/blank SMILES.
    Prints the SMILES that are dropped (capped by max_print).
    """

    counts = []
    bad_smiles_idx = []
    bad_smiles_list = []  # store SMILES strings for printing

    for idx, smi in df[smiles_col].items():
        if not isinstance(smi, str) or not smi.strip():
            bad_smiles_idx.append(idx)
            bad_smiles_list.append(smi)
            counts.append({"atom_assigned": 0, "atom_unassigned": 0, "bond_assigned": 0, "bond_unassigned": 0})
            continue

        try:
            c = count_stereocenters_from_smiles(smi)
            counts.append(c)
        except Exception:
            if drop_invalid_smiles:
                bad_smiles_idx.append(idx)
                bad_smiles_list.append(smi)
                counts.append({"atom_assigned": 0, "atom_unassigned": 0, "bond_assigned": 0, "bond_unassigned": 0})
            else:
                raise

    counts_df = pd.DataFrame(counts, index=df.index)

    # attach counts if requested
    df2 = df.join(counts_df) if keep_stereo_counts else df.copy()

    # rows that have no unassigned stereo
    ok = (counts_df["atom_unassigned"] == 0) & (counts_df["bond_unassigned"] == 0)

    # also drop invalid smiles if requested
    if drop_invalid_smiles and bad_smiles_idx:
        ok = ok & (~df2.index.isin(bad_smiles_idx))

    # ---- printing dropped SMILES ----
    if print_dropped:
        # invalid/blank
        if drop_invalid_smiles and bad_smiles_idx:
            print(f"Dropping {len(bad_smiles_idx)} invalid/blank SMILES (showing up to {max_print}):")
            for s in bad_smiles_list[:max_print]:
                print("  ", repr(s))
            if len(bad_smiles_list) > max_print:
                print(f"  ... {len(bad_smiles_list) - max_print} more")

        # unassigned stereo (valid but not ok)
        dropped_stereo_df = df2.loc[~ok, [smiles_col]].copy()
        # If drop_invalid_smiles=True, ~ok already includes invalids; remove them for this section
        if drop_invalid_smiles and bad_smiles_idx:
            dropped_stereo_df = dropped_stereo_df.loc[~dropped_stereo_df.index.isin(bad_smiles_idx)]

        n_stereo_drop = len(dropped_stereo_df)
        if n_stereo_drop:
            print(f"Dropping {n_stereo_drop} SMILES due to unassigned stereo (showing up to {max_print}):")
            for s in dropped_stereo_df[smiles_col].tolist()[:max_print]:
                print("  ", s)
            if n_stereo_drop > max_print:
                print(f"  ... {n_stereo_drop - max_print} more")

    return df2.loc[ok].reset_index(drop=True)


MOL_DIR = Path('/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/molecules')

def _is_contaminated(radname: str, s2_threshold: float) -> tuple[str, bool]:
    mol_path = MOL_DIR / radname
    if not mol_path.exists():
        return radname, False
    for out_file in mol_path.glob('DFT_elec_radical_*.out'):
        s2 = read_spin_contamination(out_file)
        if s2 is not None and s2 > s2_threshold:
            return radname, True
    return radname, False


def filter_spin_contamination(
    df: pd.DataFrame,
    s2_threshold: float = 0.85,
    radical_name_col: str = 'radical_name',
    s2_col: str = 'S2',
    n_workers: int = 16,
) -> pd.DataFrame:
    """
    Remove rows with spin contamination S**2 > s2_threshold.

    If the dataframe has an 'S2' column (written by read_files.py), filtering is
    done directly on that column — fast and independent of whether DFT files still
    exist on disk.  If the column is absent, falls back to scanning
    DFT_elec_radical_*.out files (only works before the cleanup script runs).
    """
    if s2_col in df.columns:
        mask_bad = df[s2_col].notna() & (df[s2_col] > s2_threshold)
        n_deleted = int(mask_bad.sum())
        print(f"Removed {n_deleted} rows with {s2_col} > {s2_threshold} (column-based filter)")
        return df.loc[~mask_bad].reset_index(drop=True)

    # fallback: scan DFT output files (deleted after cleanup — may find nothing)
    print(f"'S2' column not found — falling back to DFT file scan (results may be incomplete if files were cleaned up)")
    unique_names = df[radical_name_col].unique().tolist()
    contaminated = set()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_is_contaminated, name, s2_threshold): name for name in unique_names}
        for future in as_completed(futures):
            radname, bad = future.result()
            if bad:
                contaminated.add(radname)

    mask_bad = df[radical_name_col].isin(contaminated)
    n_deleted = mask_bad.sum()
    print(f"Removed {n_deleted} rows from {len(contaminated)} spin-contaminated radicals (S**2 > {s2_threshold})")
    if contaminated:
        print("  Flagged:", sorted(contaminated))

    return df.loc[~mask_bad].reset_index(drop=True)


def filter_other_radicals(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
) -> pd.DataFrame:
    """
    Remove radicals whose centre is not C, N, O, or S (e.g. P, Si, halogen-centred).
    These are too rare to train on reliably and show anomalously high errors.
    """
    from rdkit import Chem

    def radical_element(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return 'invalid'
            for atom in mol.GetAtoms():
                if atom.GetNumRadicalElectrons() > 0:
                    return atom.GetSymbol()
        except Exception:
            pass
        return 'unknown'

    elements = df[smiles_col].apply(radical_element)
    known    = {'C', 'N', 'O', 'S'}
    mask_bad = ~elements.isin(known)
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} rows with non-C/N/O/S radical centres: "
          f"{elements[mask_bad].value_counts().to_dict()}")
    return df.loc[~mask_bad].reset_index(drop=True)



_ACYLOXY_PATT       = Chem.MolFromSmarts('[O;X1][C](=O)')
_PHOSPHONATE_O_PATT = Chem.MolFromSmarts('[O;X1][P]')
_PHOSPHONATE_S_PATT = Chem.MolFromSmarts('[S;X1][P]')
_SI_PATT            = Chem.MolFromSmarts('[Si]')
_SI_S_PATT          = Chem.MolFromSmarts('[Si][S;X1]')
_N_HAL_PATT         = Chem.MolFromSmarts('[N][Cl,Br,I,F]')
_C_I_PATT           = Chem.MolFromSmarts('[C][I]')

LOCALIZATION_CSV = '/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/model_results/worst_performing_radicals_v21_localization.csv'


def filter_missing_dft(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    name_col: str = 'radical_name',
    localization_csv: str = LOCALIZATION_CSV,
) -> pd.DataFrame:
    """
    Remove entries whose DFT radical output file could not be found or parsed.

    spin_max is NaN in the localization CSV when parse_mulliken returned nothing
    for DFT_elec_radical_{h}H.out — either the file was cleaned up, never written,
    or has an unexpected format. Without a parseable radical output we cannot verify
    spin contamination, radical migration, or charge localisation, so the entry is
    dropped as unverifiable.

    Entries not present in the localization CSV at all are also removed (they were
    never checked).
    """
    loc = pd.read_csv(localization_csv, usecols=[name_col, smiles_col, 'spin_max'])
    loc = loc.drop_duplicates(subset=[name_col, smiles_col])
    merged = df.merge(loc, on=[name_col, smiles_col], how='left')
    mask_bad = merged['spin_max'].isna()
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} entries with missing or unparseable DFT radical output")
    return df.loc[~mask_bad.values].reset_index(drop=True)


def _smiles_radical_element(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                return atom.GetSymbol()
    except Exception:
        pass
    return None


def _mulliken_spin_max_elem(filepath: Path) -> str | None:
    """Element symbol with the largest Mulliken spin population in an open-shell ORCA output."""
    try:
        text = filepath.read_text()
    except Exception:
        return None
    blocks = list(re.finditer(
        r"MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n(.*?)\nSum of atomic charges",
        text, re.S,
    ))
    if not blocks:
        return None
    best_sym, best_spin = None, -float('inf')
    for line in blocks[-1].group(1).splitlines():
        m = re.match(r"\s*\d+\s+(\S+)\s*:\s*[-\d.]+\s+([-\d.]+)", line)
        if m:
            spin = float(m.group(2))
            if spin > best_spin:
                best_spin, best_sym = spin, m.group(1)
    return best_sym


def _mulliken_charges_and_spin(filepath: Path):
    """Parse per-atom charges and spin populations from an open-shell ORCA Mulliken block."""
    try:
        text = filepath.read_text()
    except Exception:
        return None, None
    blocks = list(re.finditer(
        r"MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\s*\n-+\n(.*?)\nSum of atomic charges",
        text, re.S,
    ))
    if not blocks:
        return None, None
    charges, spins = {}, {}
    for line in blocks[-1].group(1).splitlines():
        m = re.match(r"\s*(\d+)\s+\S+\s*:\s*([-\d.]+)\s+([-\d.]+)", line)
        if m:
            idx = int(m.group(1))
            charges[idx] = float(m.group(2))
            spins[idx] = float(m.group(3))
    return charges, spins


def _mulliken_charges_closed(filepath: Path):
    """Parse per-atom charges from a closed-shell ORCA Mulliken block (anion/cation)."""
    try:
        text = filepath.read_text()
    except Exception:
        return None
    blocks = list(re.finditer(
        r"MULLIKEN ATOMIC CHARGES\s*\n-+\n(.*?)\nSum of atomic charges",
        text, re.S,
    ))
    if not blocks:
        return None
    charges = {}
    for line in blocks[-1].group(1).splitlines():
        m = re.match(r"\s*(\d+)\s+\S+\s*:\s*([-\d.]+)", line)
        if m:
            charges[int(m.group(1))] = float(m.group(2))
    return charges


def _check_phosphonate_o_radical_group(args, I_tol: float = 1e-2, A_tol: float = 1e-2):
    """For one radical_name and its rows, return {row_index: ani_cos_sim or None}.

    None means unverifiable (missing molecule directory, no matching H found on
    disk, or unparseable Mulliken block) -- such rows are conservatively kept.
    """
    radname, rows = args
    mol_path = MOL_DIR / radname

    if not mol_path.exists():
        return {idx: None for idx, *_ in rows}

    # Newer molecule directories also contain DFT_elec_radical_{h}H_atom{n}.out
    # sub-calculations (per-atom resolved, not separate H-abstraction sites) --
    # match only the plain DFT_elec_radical_{h}H.out form so those don't crash
    # the int() parse below.
    h_indices = sorted(set(
        int(m.group(1))
        for p in mol_path.glob('DFT_elec_radical_*.out')
        for m in [re.fullmatch(r'DFT_elec_radical_(\d+)H', p.stem)]
        if m
    ))

    h_IA = {}
    for h in h_indices:
        try:
            I, A, _ = read_philicity(str(mol_path), h)
            h_IA[h] = (I, A)
        except Exception:
            continue

    result = {}
    for idx, I_target, A_target in rows:
        matched_h = None
        for h, (I, A) in h_IA.items():
            if abs(I - I_target) < I_tol and abs(A - A_target) < A_tol:
                matched_h = h
                break

        if matched_h is None:
            result[idx] = None
            continue

        q_rad, spin = _mulliken_charges_and_spin(mol_path / f'DFT_elec_radical_{matched_h}H.out')
        q_an = _mulliken_charges_closed(mol_path / f'DFT_elec_anion_{matched_h}H.out')
        if q_rad is None or q_an is None:
            result[idx] = None
            continue

        idxs = sorted(set(q_rad) & set(q_an))
        if not idxs:
            result[idx] = None
            continue

        s = np.array([spin[i] for i in idxs])
        dq = np.array([q_an[i] - q_rad[i] for i in idxs])
        denom = np.linalg.norm(s) * np.linalg.norm(dq)
        result[idx] = float(np.dot(s, dq) / denom) if denom > 1e-9 else None

    return result


def _check_radical_migration_group(args, I_tol: float = 1e-2, A_tol: float = 1e-2):
    """For one radical_name and its rows, return {row_index: True/False/None} migration flags.

    None means unverifiable (missing molecule directory, no matching H found on disk,
    or unparseable output) -- such rows are conservatively kept.
    """
    radname, rows = args
    mol_path = MOL_DIR / radname

    if not mol_path.exists():
        return {idx: None for idx, *_ in rows}

    # Newer molecule directories also contain DFT_elec_radical_{h}H_atom{n}.out
    # sub-calculations (per-atom resolved, not separate H-abstraction sites) --
    # match only the plain DFT_elec_radical_{h}H.out form so those don't crash
    # the int() parse below.
    h_indices = sorted(set(
        int(m.group(1))
        for p in mol_path.glob('DFT_elec_radical_*.out')
        for m in [re.fullmatch(r'DFT_elec_radical_(\d+)H', p.stem)]
        if m
    ))

    h_IA = {}
    for h in h_indices:
        try:
            I, A, _ = read_philicity(str(mol_path), h)
            h_IA[h] = (I, A)
        except Exception:
            continue

    result = {}
    for idx, smi, I_target, A_target in rows:
        matched_h = None
        for h, (I, A) in h_IA.items():
            if abs(I - I_target) < I_tol and abs(A - A_target) < A_tol:
                matched_h = h
                break

        if matched_h is None:
            result[idx] = None
            continue

        spin_max_elem = _mulliken_spin_max_elem(mol_path / f'DFT_elec_radical_{matched_h}H.out')
        smiles_elem = _smiles_radical_element(smi)

        if spin_max_elem is None or smiles_elem is None:
            result[idx] = None
        else:
            result[idx] = (spin_max_elem != smiles_elem)

    return result


def filter_radical_migrated(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    name_col: str = 'radical_name',
    I_col: str = 'I',
    A_col: str = 'A',
    n_workers: int = 24,
) -> pd.DataFrame:
    """
    Remove entries where FairChem geometry optimisation moved the radical centre
    to a different atom than specified in the SMILES (detected via Mulliken spin
    population: spin_max_elem != smiles_rad_elem in the neutral DFT output).

    Computed live, across the whole dataset, by parsing Mulliken spin populations
    directly from DFT_elec_radical_{h}H.out. The main dataset does not store
    which hydrogen index (h) was abstracted for a given row, so the correct DFT
    output file is identified by matching the row's stored (I, A) against
    read_philicity() recomputed for every h available on disk for that
    radical_name (grouped so each molecule's files are read once regardless of
    how many of its H-sites survived upstream filtering).

    Entries whose molecule directory is missing, whose (I, A) doesn't match any
    H on disk, or whose Mulliken block can't be parsed are kept (unverifiable,
    not flagged) -- consistent with the rest of the pipeline's conservative
    treatment of unverifiable data.
    """
    tasks = [
        (radname, list(zip(group.index, group[smiles_col], group[I_col], group[A_col])))
        for radname, group in df.groupby(name_col)
    ]

    migrated = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_check_radical_migration_group, t) for t in tasks]
        for future in as_completed(futures):
            migrated.update(future.result())

    flag = pd.Series(migrated).reindex(df.index)
    mask_bad = flag == True
    n_deleted = int(mask_bad.sum())
    n_unknown = int(flag.isna().sum())
    print(f"Removed {n_deleted} radical-migrated rows (live Mulliken check, spin_max_elem != SMILES elem); "
          f"{n_unknown} entries unverifiable (kept)")
    return df.loc[~mask_bad.values].reset_index(drop=True)


def filter_acyloxy(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    A_col: str = 'A',
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Remove acyloxy radicals (RCO₂•) with A < threshold (DFT beta-scission artifacts).

    FairChem geometry relaxation frequently drives beta-scission (C–C breaks,
    CO₂ departs), yielding A ≈ 0 eV instead of the expected ~3–4 eV for an intact
    acyloxy radical. The raw-data A distribution is clearly bimodal: artifacts cluster
    at A < 1.5 eV, geometrically intact entries at A > 2.5 eV, with a clean gap.
    Filtering by A < 2.0 removes only the artifacts (~12% of acyloxy entries) while
    keeping the intact ones for training.
    """
    def _is_acyloxy_radical(smi: str) -> bool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        for match in mol.GetSubstructMatches(_ACYLOXY_PATT):
            if mol.GetAtomWithIdx(match[0]).GetNumRadicalElectrons() > 0:
                return True
        return False

    is_acyloxy = df[smiles_col].apply(_is_acyloxy_radical)
    mask_bad = is_acyloxy & (df[A_col] < threshold)
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} acyloxy radical rows with A < {threshold} eV (beta-scission artifacts)")
    return df.loc[~mask_bad].reset_index(drop=True)


def filter_phosphonate_artifacts(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    philicity_col: str = 'philicity',
    threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Remove phosphonate O-radical ([O;X1][P]) and phosphorothioate S-radical
    ([S;X1][P]) entries with philicity < threshold.

    Both classes are highly electrophilic due to the electron-withdrawing P=O
    group; philicity below 2 eV is chemically unreasonable and flags a DFT
    artifact:
      - [O;X1][P]: P-O beta-scission during FairChem geometry relaxation gives
        A ≈ 0.3 eV instead of the expected ~3.5 eV.
      - [S;X1][P]: intramolecular proton transfer + P-C bond breaking in the
        anion geometry gives A ≈ 0.3 eV instead of the expected ~3 eV.
    """
    if philicity_col not in df.columns:
        print(f"Column '{philicity_col}' not found — skipping phosphonate artifact filter")
        return df

    def _matches_phosphonate(smi: str) -> bool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        return (mol.HasSubstructMatch(_PHOSPHONATE_O_PATT) or
                mol.HasSubstructMatch(_PHOSPHONATE_S_PATT))

    is_phosphonate = df[smiles_col].apply(_matches_phosphonate)
    mask_bad = is_phosphonate & (df[philicity_col] < threshold)
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} phosphonate/phosphorothioate radical rows with philicity < {threshold} eV")
    return df.loc[~mask_bad].reset_index(drop=True)



def filter_aryl_artifacts(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    A_col: str = 'A',
    I_col: str = 'I',
    A_threshold: float = 3.0,
    I_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Remove aryl C-radical entries where the anion SCF root-flips to a reducible
    substituent (N=O, C=N, halogen, etc.) instead of the aryl ring π*.

    Criterion: true aryl C-radical (C in a 6-membered ring with ≥4 aromatic atoms)
    AND A > 3.0 eV AND I < 10.0 eV.

    Legitimate ultra-electron-deficient aryl radicals (e.g. pyridinium N-oxide + CN)
    have both elevated I (>10 eV) and elevated A; root-flipping artifacts have
    normal I (<10 eV) but anomalously high A (>3 eV) from the added electron
    going to a substituent π* or σ* state.
    """
    if A_col not in df.columns or I_col not in df.columns:
        print(f"Columns '{A_col}'/'{I_col}' not found — skipping aryl artifact filter")
        return df

    def _is_aryl_c_radical(smi: str) -> bool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) != 6:
                continue
            n_arom = sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetIsAromatic())
            if n_arom < 4:
                continue
            for idx in ring:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetNumRadicalElectrons() > 0 and atom.GetSymbol() == 'C':
                    return True
        return False

    is_aryl = df[smiles_col].apply(_is_aryl_c_radical)
    mask_bad = is_aryl & (df[A_col] > A_threshold) & (df[I_col] < I_threshold)
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} aryl C-radical rows with A > {A_threshold} eV and I < {I_threshold} eV (root-flipping to substituent)")
    return df.loc[~mask_bad].reset_index(drop=True)


def filter_halothiophene_artifacts(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    A_col: str = 'A',
    A_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Remove 5-membered-thiophene-ring C-radicals bearing a ring halogen substituent
    with A > A_threshold.

    Same mechanism as filter_aryl_artifacts (anion SCF root-flips to an easily
    reducible substituent instead of the ring SOMO) but for 5-membered S-heterocycles,
    which filter_aryl_artifacts excludes (it requires a 6-membered ring). Confirmed
    via direct Mulliken parsing: ani_cos_sim is shifted positive (-0.53 to -0.67 vs.
    ~-1 for a clean calc) and the largest anion charge shift lands on the ring S and
    the halogen, not the radical center. The broad pattern alone isn't predictive
    (716 entries, 2.0% error>0.5, below the 3.5% dataset baseline) -- only the A>3.0
    tail is enriched (40 entries, 30.0% error>0.5, ~8.6x baseline).
    """
    if A_col not in df.columns:
        print(f"Column '{A_col}' not found — skipping halothiophene artifact filter")
        return df

    def _is_halothiophene_ring_radical(smi: str) -> bool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) != 5:
                continue
            if sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetSymbol() == 'S') != 1:
                continue
            if not any(mol.GetAtomWithIdx(idx).GetNumRadicalElectrons() > 0 for idx in ring):
                continue
            if any(
                nbr.GetSymbol() in ('Cl', 'Br', 'I', 'F')
                for idx in ring for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            ):
                return True
        return False

    is_match = df[smiles_col].apply(_is_halothiophene_ring_radical)
    mask_bad = is_match & (df[A_col] > A_threshold)
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} halothiophene ring-radical rows with A > {A_threshold} eV (anion root-flip to ring S/halogen)")
    return df.loc[~mask_bad].reset_index(drop=True)


def filter_si_cation_artifacts(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    I_col: str = 'I',
    A_col: str = 'A',
    I_threshold: float = 11.0,
    A_threshold_si_s: float = 1.5,
) -> pd.DataFrame:
    """
    Remove two classes of Si-related DFT artifacts:
      1. [Si] in molecule + I > I_threshold: cation SCF removes electron from
         Si-O sigma* instead of the SOMO (root-flipping), yielding anomalously
         high I. Threshold 11 eV cleanly separates from legitimate radicals.
      2. [Si][S•] (Si directly bonded to S-radical) + A < A_threshold_si_s:
         anion SCF delocalizes into Si-O bond rather than the S SOMO, giving
         anomalously low A (~0.95-1.4 eV). Catches Si-S entries that escape
         the I threshold because their I is moderate (7.6-8.3 eV).
    """
    def _has_si(smi):
        mol = Chem.MolFromSmiles(smi)
        return mol is not None and mol.HasSubstructMatch(_SI_PATT)

    def _has_si_s(smi):
        mol = Chem.MolFromSmiles(smi)
        return mol is not None and mol.HasSubstructMatch(_SI_S_PATT)

    has_si   = df[smiles_col].apply(_has_si)
    has_si_s = df[smiles_col].apply(_has_si_s)

    mask_high_i = has_si   & (df[I_col] > I_threshold)
    mask_si_s   = has_si_s & (df[A_col] < A_threshold_si_s)
    mask_bad = mask_high_i | mask_si_s

    n_high_i = int(mask_high_i.sum())
    n_si_s   = int((mask_si_s & ~mask_high_i).sum())
    print(f"Removed {mask_bad.sum()} Si artifact rows: "
          f"{n_high_i} [Si]+I>{I_threshold} (cation root-flip), "
          f"{n_si_s} [Si][S•]+A<{A_threshold_si_s} (anion delocalization)")
    return df.loc[~mask_bad].reset_index(drop=True)


def filter_halogen_cation_artifacts(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    I_col: str = 'I',
    I_threshold_n_hal: float = 9.5,
    I_threshold_c_i: float = 9.0,
) -> pd.DataFrame:
    """
    Remove cation root-flipping artifacts caused by N-halogen or C-iodide bonds.

    When the cation SCF removes an electron from a N-X or C-I sigma* orbital
    instead of the SOMO, the computed I is anomalously high. Two sub-classes:
      - [N][Cl,Br,I,F] + I > 9.5 eV: N-halogen bond lower IE than SOMO
      - [C][I] + I > 9.0 eV: C-I bond lower IE than SOMO (slightly lower
        threshold because C-I sigma* ionizes more easily than N-X)
    """
    def _has_n_hal(smi):
        mol = Chem.MolFromSmiles(smi)
        return mol is not None and mol.HasSubstructMatch(_N_HAL_PATT)

    def _has_c_i(smi):
        mol = Chem.MolFromSmiles(smi)
        return mol is not None and mol.HasSubstructMatch(_C_I_PATT)

    has_n_hal = df[smiles_col].apply(_has_n_hal)
    has_c_i   = df[smiles_col].apply(_has_c_i)

    mask_n_hal = has_n_hal & (df[I_col] > I_threshold_n_hal)
    mask_c_i   = has_c_i   & (df[I_col] > I_threshold_c_i)
    mask_bad   = mask_n_hal | mask_c_i

    n_n_hal = int(mask_n_hal.sum())
    n_c_i   = int((mask_c_i & ~mask_n_hal).sum())
    print(f"Removed {mask_bad.sum()} halogen cation root-flip rows: "
          f"{n_n_hal} [N][X]+I>{I_threshold_n_hal}, "
          f"{n_c_i} [C][I]+I>{I_threshold_c_i} (net new)")
    return df.loc[~mask_bad].reset_index(drop=True)


def filter_n_radical_peroxide_artifacts(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    A_col: str = 'A',
    A_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove anion root-flipping artifacts where an N-radical sits directly on
    one oxygen of an O-O (peroxide) linkage: [N*]-O-O-R.

    Peroxide O-O sigma* is an easily-reduced orbital; the anion SCF frequently
    puts the added electron there instead of the N SOMO, giving anomalously
    high A. Confirmed via direct Mulliken parsing of DFT output files: ani_cos_sim
    is strongly positive (+0.2 to +0.5, vs. ~-1 for a correct calculation), and
    the largest anion charge shift lands on a peroxide oxygen, not the radical
    center. Threshold A>1.5 eV catches the artifact cluster (71.9% precision for
    error>0.5 against worst_performing_radicals_v26.csv).
    """
    def _has_n_radical_oo(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetNumRadicalElectrons() > 0:
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == 'O':
                        for nbr2 in nbr.GetNeighbors():
                            if nbr2.GetIdx() != atom.GetIdx() and nbr2.GetSymbol() == 'O':
                                return True
        return False

    if A_col not in df.columns:
        print(f"Column '{A_col}' not found — skipping N-radical peroxide filter")
        return df

    is_n_rad_oo = df[smiles_col].apply(_has_n_radical_oo)
    mask_bad = is_n_rad_oo & (df[A_col] > A_threshold)
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} N-radical peroxide rows ([N*]-O-O-R) with A > {A_threshold} eV (anion root-flip)")
    return df.loc[~mask_bad].reset_index(drop=True)


_EXCLUDED_RADICALS = {
    # (radical_name, radical_smiles) -- hand-identified bad radicals confirmed by
    # manual investigation that don't fit cleanly into any sweeping structural
    # filter. Either the artifact is borderline (narrowly missing a threshold
    # tuned for precision elsewhere) or the broader structural class is mostly
    # legitimate, so a wholesale filter would remove too much good data.
    ('p5_9878', '[C]#CCCCN'),
    # Terminal alkynyl (sp) C-radical, I=8.40 eV vs. ~10-12 eV typical for this
    # class. Direct Mulliken check is too clean for a root-flip (cat_cos_sim=0.86,
    # ani_cos_sim=-0.77), so likely a geometry-dependent through-space stabilization
    # from the amine 4 atoms down the chain rather than SCF state-switching --
    # not a pattern worth filtering wholesale since most terminal-alkynyl radicals
    # are legitimate (high I from sp character) and the model under-predicts them.
    ('p5_7865', 'CN(C)[C@@H]([CH]O)I'),
    ('p5_1435', 'C[C](C)CI'),
    # Both beta to a C-I bond (same cation root-flip mechanism as
    # filter_halogen_cation_artifacts's [C][I] check) but I=8.88 eV, just under
    # the 9.0 eV threshold tuned for precision there. Lowering that threshold to
    # catch these two costs 51 false positives (3.8% precision) for the dataset
    # as a whole, so they are excluded individually instead.
    ('p5_167417', 'N=C([N]Cl)c1[nH]n(CCO)c2c1ccc1nncc21'),
    # Same mechanism as filter_halogen_cation_artifacts's N-halogen check
    # ([N][Cl,Br,I,F]) but I=9.45 eV, just under the 9.5 eV threshold there.
    # Same borderline-threshold situation as the two C-I cases above.
    ('p5_28067', 'CN1[C]=C(C=C1C(=O)O)I'),
    # Same C-I cation root-flip mechanism, I=8.98 eV (again just under the
    # 9.0 eV threshold). Confirmed via Mulliken: the spin-max ring carbon
    # doesn't even appear among the top anion charge-responding atoms --
    # the added electron goes almost entirely to the iodine instead
    # (dq=-0.26 on I vs the spin-max carbon not making the top 6).
}


def filter_individual_exclusions(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    name_col: str = 'radical_name',
    excluded: set = None,
) -> pd.DataFrame:
    """
    Remove specific hand-identified radicals from _EXCLUDED_RADICALS, keyed by
    (radical_name, radical_smiles). See that constant for per-entry rationale.
    """
    if excluded is None:
        excluded = _EXCLUDED_RADICALS
    keys = list(zip(df[name_col], df[smiles_col]))
    mask_bad = pd.Series([k in excluded for k in keys], index=df.index)
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} individually-excluded radicals: {sorted(set(keys) & excluded)}")
    return df.loc[~mask_bad].reset_index(drop=True)


def filter_ring_sulfurane_artifacts(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
) -> pd.DataFrame:
    """
    Remove molecules containing a chemically invalid ring sulfur: a ring S with
    >=3 connections that has a double bond to a ring carbon (explicit_valence=4,
    bonded entirely to C/N, not to O as in a normal sulfoxide/sulfone).

    Confirmed present even in original_smiles (i.e. baked into the parent
    molecule library, not introduced by radical generation, FairChem, or DFT) --
    likely from an upstream heteroatom-substitution step that swapped a ring CH
    for S without correcting for sulfur's normal divalent ring connectivity
    (cf. thiophene S, which only takes 2 ring bonds). This is not a wrong-SCF-state
    artifact; the structure itself ("sulfurane"/S-ylide bonded only to C/N) is not
    standard, stable organosulfur chemistry, so it is removed outright regardless
    of I/A, the same way filter_other_radicals removes non-C/N/O/S radical centres.

    Confirmed error enrichment: 950 entries in worst_performing_radicals_v26.csv,
    20.9% with error>0.5 vs. 3.9% baseline (~5.4x), median error ~2x baseline.
    """
    def _has_ring_sulfurane(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'S' or not atom.IsInRing() or atom.GetDegree() < 3:
                continue
            if any(
                mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondTypeAsDouble() == 2
                and nbr.GetSymbol() == 'C'
                for nbr in atom.GetNeighbors()
            ):
                return True
        return False

    mask_bad = df[smiles_col].apply(_has_ring_sulfurane)
    n_deleted = int(mask_bad.sum())
    print(f"Removed {n_deleted} ring-sulfurane rows (invalid hypervalent ring S=C, not a sulfoxide/sulfone)")
    return df.loc[~mask_bad].reset_index(drop=True)


def filter_phosphonate_o_radical_artifacts(
    df: pd.DataFrame,
    smiles_col: str = 'radical_smiles',
    name_col: str = 'radical_name',
    I_col: str = 'I',
    A_col: str = 'A',
    ani_cos_sim_threshold: float = -0.75,
    n_workers: int = 24,
) -> pd.DataFrame:
    """
    Remove phosphonate O-radical ([O;X1][P]) entries where the anion SCF
    root-flips to the P-O or P-C bond instead of the O-radical SOMO.

    Computed live, like filter_radical_migrated, by matching each row to its
    DFT output file via (I, A) and parsing Mulliken charges/spin directly --
    the main dataset doesn't store which hydrogen was abstracted, so the
    correct H is identified the same way as in filter_radical_migrated.

    Re-validated on the full live-computed phosphonate population (523 entries
    in worst_performing_radicals_v26.csv): ani_cos_sim > -0.75 isolates 10
    entries at 60% precision for error>0.5, consistent with the original n=9
    finding (66.7%) that motivated this threshold -- not a fluke of the small
    sample size.

    Entries whose molecule directory is missing, whose (I, A) doesn't match any
    H on disk, or whose Mulliken block can't be parsed are kept (unverifiable).
    """
    def _has_o_p_rad(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        return any(mol.GetAtomWithIdx(x[0]).GetNumRadicalElectrons() > 0
                   for x in mol.GetSubstructMatches(_PHOSPHONATE_O_PATT))

    is_o_p_rad = df[smiles_col].apply(_has_o_p_rad)
    if not is_o_p_rad.any():
        print("Removed 0 phosphonate O-radical artifact rows")
        return df

    candidates = df.loc[is_o_p_rad]
    tasks = [
        (radname, list(zip(group.index, group[I_col], group[A_col])))
        for radname, group in candidates.groupby(name_col)
    ]

    cos_sims = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_check_phosphonate_o_radical_group, t) for t in tasks]
        for future in as_completed(futures):
            cos_sims.update(future.result())

    cos_sim = pd.Series(cos_sims).reindex(df.index)
    mask_bad = (cos_sim > ani_cos_sim_threshold).fillna(False)
    n_deleted = int(mask_bad.sum())
    n_unknown = int(is_o_p_rad.sum() - cos_sim.notna().sum())
    print(f"Removed {n_deleted} phosphonate O-radical rows with ani_cos_sim > {ani_cos_sim_threshold} (anion root-flip); "
          f"{n_unknown} phosphonate entries unverifiable (kept)")
    return df.loc[~mask_bad.values].reset_index(drop=True)


def filter_unphysical_energies(
    df: pd.DataFrame,
    I_col: str = 'I',
    A_col: str = 'A',
    I_min: float = 5.0,
    I_max: float = 15.0,
    I_max_cation: float = 35.0,
    A_min: float = -2.0,
    smiles_col: str = 'radical_smiles',
) -> pd.DataFrame:
    """
    Remove rows where the DFT-derived ionization energy or electron affinity is
    physically impossible, indicating SCF convergence to a wrong electronic state.

    Observed failure modes:
      - Anion SCF diverges to a wrong root → A drops to -19 or -22 eV while I is
        unchanged (the radical and cation energies are identical between runs).
      - Cation or combined failure → I jumps to 18+ eV or falls below 5 eV.

    Thresholds are conservative — all known legitimate organic radicals lie well
    within [5, 15] eV for I and above -2 eV for A.

    For radical cations, I is the second ionization energy (radical cation →
    dication), which is legitimately 15-30 eV — a separate upper bound
    (I_max_cation=35.0) is applied to those rows.
    """
    if I_col not in df.columns or A_col not in df.columns:
        print(f"Columns '{I_col}'/'{A_col}' not found — skipping unphysical energy filter")
        return df

    is_cation = df[smiles_col].str.contains(r"\+", na=False) if smiles_col in df.columns else pd.Series(False, index=df.index)
    effective_I_max = is_cation.map({True: I_max_cation, False: I_max})

    mask_bad = (
        (df[I_col] < I_min) |
        (df[I_col] > effective_I_max) |
        (df[A_col] < A_min)
    )
    n_neutral_bad = int((mask_bad & ~is_cation).sum())
    n_cation_bad = int((mask_bad & is_cation).sum())
    n_deleted = int(mask_bad.sum())
    breakdown = {
        f'{I_col} < {I_min}': int((df[I_col] < I_min).sum()),
        f'{I_col} > {I_max} (neutral)': int((~is_cation & (df[I_col] > I_max)).sum()),
        f'{I_col} > {I_max_cation} (cation)': int((is_cation & (df[I_col] > I_max_cation)).sum()),
        f'{A_col} < {A_min}': int((df[A_col] < A_min).sum()),
    }
    print(f"Removed {n_deleted} rows with unphysical I/A values "
          f"(neutral: {n_neutral_bad}, cation: {n_cation_bad}): {breakdown}")
    return df.loc[~mask_bad].reset_index(drop=True)


if __name__ == '__main__':

    # Input is compile_and_filter.py's output (the lightly-filtered, freshly
    # recompiled batch data -- it already applies most of these filters, but
    # skips filter_missing_dft/filter_radical_migrated/
    # filter_phosphonate_o_radical_artifacts since brand-new radicals don't
    # have matched DFT/localization data yet). Output stays the canonical
    # _with_clusters path so every modeling script picks up the refresh
    # without code changes. Previously this read+wrote the same canonical
    # file in place, which only re-applied filters to already-canonical data
    # and never pulled in newer compiled batches.
    df = pd.read_csv('/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/all_radicals/all_philicities_compiled.csv')

    df = filter_unassigned_stereo_rows(df)
    df = filter_other_radicals(df)
    df = filter_individual_exclusions(df)
    df = filter_ring_sulfurane_artifacts(df)
    df = filter_spin_contamination(df)
    df = filter_unphysical_energies(df)
    df = filter_acyloxy(df)
    df = filter_phosphonate_artifacts(df)
    df = filter_aryl_artifacts(df)
    df = filter_halothiophene_artifacts(df)
    df = filter_si_cation_artifacts(df)
    df = filter_halogen_cation_artifacts(df)
    df = filter_n_radical_peroxide_artifacts(df)
    df = filter_radical_migrated(df)
    df = filter_phosphonate_o_radical_artifacts(df)
    df = filter_philicity(df)

    df.to_csv('/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/all_radicals/all_philicities_compiled_with_clusters.csv', index=False)
