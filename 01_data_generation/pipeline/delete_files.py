"""Per-molecule cleanup, run at the end of each molecule's job chain.

`submit_philicities.py` submits one of these per molecule with an `afterany`
dependency on that molecule's `read_files_*H.sh` jobs, so it fires once the
philicities have been extracted (or once the attempts have failed).

Policy is keep-list based, matching the convention already used by
`clean_up.py`: anything not explicitly kept is junk. Every `.out` file is
kept, so `read_files.py` can always be re-run against a cleaned directory --
nothing this script deletes is needed to recover a philicity.

What actually costs space is ORCA scratch. On an uncleaned single-point
directory `.gbw` + `.densities` are ~81% of the footprint (12.1 MB of 15 MB)
while the `.out` files that carry the answer are 0.68 MB.
"""

from __future__ import annotations

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path

DEFAULT_MOLECULES_ROOT = Path(
    "/insomnia001/depts/tekle_smith/users/MKL/project_5/data_generation/molecules"
)

# Checked before KEEP_GLOBS, so these win over the broad "*.xyz" keep below.
# GOAT/xtb write a lot of intermediate geometries; only the global minimum and
# the per-radical geometries are worth keeping.
JUNK_OVERRIDE_GLOBS = (
    "goat.finalensemble*",
    "goat.[0-9]*",
    "goat.goat.*",
    "goat.xyz",
    "*_trj.xyz",
    "*xtbrestart",
    "xtbhess.xyz",
)

# Kept unconditionally. Checked before any delete rule, so a keep always wins.
KEEP_GLOBS = (
    # every log/output: DFT_elec_*.out, fairchem_*.xyz.out, goat.out, xtb.out,
    # splitgoat.out, read_files_*H.out, submit_philicities.out, delete_files.out
    "*.out",
    # geometries needed to re-run a DFT step without redoing FairChem/GOAT.
    # Broad by design: every .xyz that is not caught by JUNK_OVERRIDE_GLOBS
    # above is a real structure (geom_*H.xyz, *H.xyz, goat.globalminimum.xyz,
    # xtbopt.xyz, the p5_*.xyz input, the cation pipeline's radical*.xyz), and
    # they are all a few KB.
    "*.xyz",
    # tiny, and worth keeping for reproducibility
    "*.inp",
    "*.sh",
    "*.csv",
    "*.py",
    "*.json",
    "*.md",
)

# Deleted when not kept above. Listed explicitly rather than "everything else"
# so an unrecognised file is reported instead of silently removed.
DELETE_GLOBS = (
    # ORCA scratch -- the bulk of the footprint
    "*.gbw", "*.densities", "*.densitiesinfo", "*.bibtex", "*.property.txt",
    "*.tmp", "*.opt", "*.engrad", "*.hess", "*.cis", "*.nbo", "*.47",
    "*.molden*", "*.ges", "*.prop", "*.uco", "*.uno", "*.unso", "*.qro",
    "*.lastscf", "*.cpcm*", "*.scfp", "*.scfr", "*.log",
    # per-atom fragment files ORCA emits for some jobs
    "*_atom[0-9]*",
    # GOAT conformer ensembles: large, and nothing downstream reads them
    # (split_radicals.py uses goat.globalminimum.xyz only)
    "goat.finalensemble*", "goat.[0-9]*",
    # xtb scratch
    "xtbrestart", "xtbtopo.mol", "wbo", "charges", ".xtboptok",
    "energy", "gradient", "xtbhess.xyz",
)


def classify(name: str) -> str:
    """Return 'keep', 'delete', or 'unknown' for a filename."""
    if any(fnmatch(name, g) for g in JUNK_OVERRIDE_GLOBS):
        return "delete"
    if any(fnmatch(name, g) for g in KEEP_GLOBS):
        return "keep"
    if any(fnmatch(name, g) for g in DELETE_GLOBS):
        return "delete"
    return "unknown"


def resolve_target(molname: str, molecules_root: Path) -> Path:
    """Resolve molecules_root/molname, refusing anything that escapes the root."""
    if not molname or molname in (".", "..") or "/" in molname or "\\" in molname:
        raise ValueError(f"refusing unsafe molname: {molname!r}")

    root = molecules_root.resolve()
    target = (root / molname).resolve()

    # containment check -- a symlinked molname must not redirect the delete
    if target != root and root not in target.parents:
        raise ValueError(f"refusing target outside molecules root: {target}")
    if target == root:
        raise ValueError("refusing to operate on the molecules root itself")
    return target


def clean_molecule(molname: str, molecules_root: Path, dry_run: bool,
                   delete_unknown: bool) -> int:
    target = resolve_target(molname, molecules_root)

    if not target.is_dir():
        print(f"nothing to do: {target} is not a directory")
        return 0

    freed = kept = removed = 0
    unknown: list[str] = []

    # top level only -- these directories are flat, and this avoids ever
    # recursing into something unexpected
    for path in sorted(target.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue

        verdict = classify(path.name)
        if verdict == "unknown":
            unknown.append(path.name)
            if not delete_unknown:
                kept += 1
                continue
            verdict = "delete"

        if verdict == "keep":
            kept += 1
            continue

        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue

        if dry_run:
            print(f"[DRY RUN] would delete {path.name} ({size / 1048576:.2f} MB)")
            freed += size
            removed += 1
            continue

        try:
            path.unlink()
        except FileNotFoundError:
            continue  # vanished between listing and unlink
        except OSError as exc:
            print(f"could not delete {path.name}: {exc}")
            continue

        freed += size
        removed += 1

    if unknown:
        action = "deleted" if delete_unknown else "kept (use --delete-unknown to remove)"
        print(f"unrecognised files {action}: {', '.join(sorted(set(unknown))[:20])}")

    verb = "would free" if dry_run else "freed"
    print(f"{molname}: {removed} files removed, {kept} kept, {verb} {freed / 1048576:.2f} MB")
    return freed


def parse_args():
    p = argparse.ArgumentParser(
        description="Delete ORCA/xtb/GOAT scratch for a finished molecule, keeping all .out files."
    )
    p.add_argument("--molname", required=True,
                   help="molecule directory name, e.g. p5_1234")
    p.add_argument("--molecules_root", default=str(DEFAULT_MOLECULES_ROOT),
                   help="directory containing the per-molecule folders")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be deleted without deleting")
    p.add_argument("--delete-unknown", action="store_true",
                   help="also delete files matching neither keep nor delete list")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        clean_molecule(args.molname, Path(args.molecules_root),
                       args.dry_run, args.delete_unknown)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
