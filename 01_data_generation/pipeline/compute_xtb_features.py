from __future__ import annotations

"""
Computes cheap, inference-time-computable electronic descriptors per unique
radical SMILES using GFN2-xTB (semi-empirical, NOT full DFT -- this is the
whole point: these features must be computable for a novel molecule without
running the expensive ORCA calculation the ML model exists to avoid).

For each radical:
  1. RDKit ETKDGv3 embed -> GFN2-xTB geometry optimization of the NEUTRAL
     radical (charge 0, uhf 1 = doublet).
  2. Single points at the SAME optimized geometry for the cation (charge +1,
     uhf 0 = N-1 electrons) and anion (charge -1, uhf 0 = N+1 electrons) --
     a frozen-geometry finite-difference approach, the standard cheap way to
     get condensed Fukui indices.
  3. Parse HOMO/LUMO/gap (neutral run) and per-atom Mulliken charges (all
     three runs), condensed to the radical-center atom:
       fukui_plus  = q_anion[rc]  - q_neutral[rc]
       fukui_minus = q_neutral[rc] - q_cation[rc]
       fukui_zero  = (q_anion[rc] - q_cation[rc]) / 2

Results are cached to a CSV keyed by canonical SMILES, written incrementally
so a long run can be resumed (already-cached SMILES are skipped).
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

XTB_BIN = "/insomnia001/depts/tekle_smith/users/softwares/miniconda/bin/xtb"
DEFAULT_CACHE_CSV = Path(
    "/insomnia001/depts/tekle_smith/users/MKL/project_5/data_analysis/all_radicals/xtb_features_cache.csv"
)
CACHE_COLUMNS = [
    "canonical_smiles", "homo", "lumo", "gap",
    "fukui_plus", "fukui_minus", "fukui_zero",
    "xtb_status", "xtb_runtime_sec",
]
TIMEOUT_SEC = 120

# xtb defaults to one OpenMP thread per available core (160 on this node). With
# N worker processes each spawning an unconstrained xtb call, that's massive
# oversubscription (N x 160 threads contending for 160 cores) -- observed
# directly: a single-atom-radical xtb --opt that normally finishes in <1s took
# many CPU-minutes once 4 concurrent xtb calls were each trying to grab all
# cores. Pin each xtb subprocess to 1 thread; ProcessPoolExecutor's n_workers
# is what actually controls parallelism.
_XTB_ENV = dict(os.environ)
_XTB_ENV.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_radical_center_index(mol) -> Optional[int]:
    rad_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if len(rad_atoms) != 1:
        return None
    return rad_atoms[0]


def embed_3d(canon_smiles: str, seed: int = 0xF00D) -> Optional["Chem.Mol"]:
    mol = Chem.MolFromSmiles(canon_smiles, sanitize=True)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return None
    return mol_h


def run_xtb(xyz_path: Path, workdir: Path, charge: int, uhf: int, opt: Optional[str] = None):
    cmd = [XTB_BIN, str(xyz_path), "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf)]
    if opt:
        cmd += ["--opt", opt]
    try:
        result = subprocess.run(
            cmd, cwd=str(workdir), capture_output=True, text=True, timeout=TIMEOUT_SEC, env=_XTB_ENV,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if result.returncode != 0:
        return False, "xtb_nonzero_exit"
    return True, result.stdout


def parse_homo_lumo(stdout: str):
    homo = lumo = None
    for line in stdout.splitlines():
        if "(HOMO)" in line:
            toks = line.split()
            try:
                homo = float(toks[-2])
            except (ValueError, IndexError):
                pass
        elif "(LUMO)" in line:
            toks = line.split()
            try:
                lumo = float(toks[-2])
            except (ValueError, IndexError):
                pass
    return homo, lumo


def parse_charges(workdir: Path) -> Optional[np.ndarray]:
    charges_path = workdir / "charges"
    if not charges_path.exists():
        return None
    try:
        arr = np.loadtxt(charges_path)
        return np.atleast_1d(arr)
    except Exception:
        return None


def compute_one(smiles: str) -> dict:
    t0 = time.time()
    canon = canonicalize_smiles(smiles)
    if canon is None:
        return dict(canonical_smiles=smiles, xtb_status="invalid_smiles")

    mol = Chem.MolFromSmiles(canon)
    rc_idx = get_radical_center_index(mol)
    if rc_idx is None:
        return dict(canonical_smiles=canon, xtb_status="no_radical_center")

    mol_h = embed_3d(canon)
    if mol_h is None:
        return dict(canonical_smiles=canon, xtb_status="embed_failed")

    tmpdir = Path(tempfile.mkdtemp(prefix="xtbfeat_"))
    try:
        neutral_dir = tmpdir / "neutral"
        neutral_dir.mkdir()
        xyz0 = neutral_dir / "mol.xyz"
        Chem.MolToXYZFile(mol_h, str(xyz0))

        ok, out = run_xtb(xyz0, neutral_dir, charge=0, uhf=1, opt="tight")
        if not ok:
            # one retry after a cheap MMFF pre-relaxation, for difficult starting geometries
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
                Chem.MolToXYZFile(mol_h, str(xyz0))
            except Exception:
                pass
            ok, out = run_xtb(xyz0, neutral_dir, charge=0, uhf=1, opt="tight")
            if not ok:
                return dict(canonical_smiles=canon, xtb_status=f"neutral_failed:{out}")

        homo, lumo = parse_homo_lumo(out)
        q_neutral = parse_charges(neutral_dir)
        opt_xyz = neutral_dir / "xtbopt.xyz"
        if not opt_xyz.exists() or q_neutral is None or homo is None or lumo is None:
            return dict(canonical_smiles=canon, xtb_status="neutral_parse_failed")
        if rc_idx >= len(q_neutral):
            return dict(canonical_smiles=canon, xtb_status="rc_index_out_of_range")

        cation_dir = tmpdir / "cation"
        cation_dir.mkdir()
        shutil.copy(opt_xyz, cation_dir / "mol.xyz")
        ok_cat, _ = run_xtb(cation_dir / "mol.xyz", cation_dir, charge=1, uhf=0)
        q_cation = parse_charges(cation_dir) if ok_cat else None

        anion_dir = tmpdir / "anion"
        anion_dir.mkdir()
        shutil.copy(opt_xyz, anion_dir / "mol.xyz")
        ok_an, _ = run_xtb(anion_dir / "mol.xyz", anion_dir, charge=-1, uhf=0)
        q_anion = parse_charges(anion_dir) if ok_an else None

        if (
            q_cation is None or q_anion is None
            or rc_idx >= len(q_cation) or rc_idx >= len(q_anion)
        ):
            return dict(
                canonical_smiles=canon, homo=homo, lumo=lumo, gap=lumo - homo,
                xtb_status="charged_state_failed",
            )

        fukui_plus = float(q_anion[rc_idx] - q_neutral[rc_idx])
        fukui_minus = float(q_neutral[rc_idx] - q_cation[rc_idx])
        fukui_zero = float((q_anion[rc_idx] - q_cation[rc_idx]) / 2.0)

        return dict(
            canonical_smiles=canon, homo=homo, lumo=lumo, gap=lumo - homo,
            fukui_plus=fukui_plus, fukui_minus=fukui_minus, fukui_zero=fukui_zero,
            xtb_status="ok", xtb_runtime_sec=round(time.time() - t0, 2),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def load_cached_smiles(cache_csv: Path) -> set[str]:
    if not cache_csv.exists():
        return set()
    try:
        existing = pd.read_csv(cache_csv, usecols=["canonical_smiles"])
        return set(existing["canonical_smiles"].dropna().unique().tolist())
    except Exception:
        return set()


def append_rows(cache_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in CACHE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[CACHE_COLUMNS]
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, mode="a", header=not cache_csv.exists(), index=False)


def compute_features_for_smiles_list(
    smiles_list: list[str],
    cache_csv: Path = DEFAULT_CACHE_CSV,
    n_workers: int = 8,
    flush_every: int = 50,
) -> None:
    """Compute (or skip, if already cached) xtb features for each unique
    canonical SMILES in smiles_list, appending incrementally to cache_csv so
    the run is resumable."""
    already_done = load_cached_smiles(cache_csv)
    canon_list = []
    seen = set()
    for s in smiles_list:
        c = canonicalize_smiles(s)
        if c is None or c in seen or c in already_done:
            continue
        seen.add(c)
        canon_list.append(c)

    print(f"{len(smiles_list)} input SMILES -> {len(canon_list)} unique, uncached canonical radicals to compute "
          f"({len(already_done)} already cached).")
    if not canon_list:
        return

    t_start = time.time()
    n_done = 0
    buffer = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(compute_one, s): s for s in canon_list}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                row = dict(canonical_smiles=futures[fut], xtb_status=f"worker_exception:{e}")
            buffer.append(row)
            n_done += 1
            if len(buffer) >= flush_every:
                append_rows(cache_csv, buffer)
                buffer = []
            if n_done % 100 == 0 or n_done == len(canon_list):
                elapsed = time.time() - t_start
                print(f"  {n_done}/{len(canon_list)} done ({elapsed:.0f}s elapsed, "
                      f"{elapsed / max(n_done, 1):.2f}s/mol avg)")
    append_rows(cache_csv, buffer)
    print(f"Done. Cache at {cache_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", type=str, default=None,
                         help="Newline-delimited file of radical SMILES (one per line).")
    parser.add_argument("--input-csv", type=str, default=None,
                         help="Alternative to --smiles-file: a CSV with a SMILES column.")
    parser.add_argument("--smiles-col", type=str, default="radical_smiles")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_CACHE_CSV))
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N unique SMILES (testing).")
    args = parser.parse_args()

    if args.smiles_file:
        smiles_list = [l.strip() for l in Path(args.smiles_file).read_text().splitlines() if l.strip()]
    elif args.input_csv:
        smiles_list = pd.read_csv(args.input_csv)[args.smiles_col].dropna().tolist()
    else:
        raise SystemExit("Provide either --smiles-file or --input-csv")

    if args.limit:
        smiles_list = smiles_list[: args.limit]

    compute_features_for_smiles_list(
        smiles_list, cache_csv=Path(args.output_csv), n_workers=args.n_workers,
    )
