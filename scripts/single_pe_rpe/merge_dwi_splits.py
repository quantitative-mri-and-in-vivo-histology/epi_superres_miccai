"""Merge split DWI acquisitions (A/B/C) into single files per PE direction.

Reads from source/sub-*/dwi/ where acquisitions are split across A/B/C runs,
and merges NIfTI volumes, bval, bvec, and JSON sidecar into single files
written to raw/sub-*/dwi/.
"""

import argparse
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


# Pattern to identify the split letter (A, B, C, ...) in the acq tag
# e.g. "acq-ninEPdiffvb104mb20A" -> group(1)="acq-ninEPdiffvb104mb20", group(2)="A"
SPLIT_PATTERN = re.compile(r"(acq-[a-zA-Z0-9]+?)([A-Z])(_dir-)")


def clean_filename(filename: str) -> str:
    """Remove acq-<tag> and ses-<tag> from a BIDS filename."""
    filename = re.sub(r"_ses-[a-zA-Z0-9]+", "", filename)
    filename = re.sub(r"_acq-[a-zA-Z0-9]+", "", filename)
    return filename


def get_group_key(filename: str) -> str | None:
    """Extract the group key by removing the split letter, acq, and ses tags."""
    m = SPLIT_PATTERN.search(filename)
    if m:
        merged = SPLIT_PATTERN.sub(r"\1\3", filename)
        return clean_filename(merged)
    return None


def get_split_letter(filename: str) -> str | None:
    """Extract the split letter (A, B, C, ...) from the filename."""
    m = SPLIT_PATTERN.search(filename)
    if m:
        return m.group(2)
    return None


def find_split_groups(dwi_dir: Path) -> dict[str, list[Path]]:
    """Group NIfTI files by their merged name (without split letter).

    Returns
    -------
    dict
        Mapping from merged filename to sorted list of split NIfTI paths.
    """
    groups: dict[str, list[Path]] = {}
    for nii in sorted(dwi_dir.glob("*.nii.gz")):
        key = get_group_key(nii.name)
        if key is None:
            continue
        groups.setdefault(key, []).append(nii)

    # Sort each group by split letter
    for key in groups:
        groups[key].sort(key=lambda p: get_split_letter(p.name) or "")

    return groups


def merge_nifti(paths: list[Path], output: Path) -> None:
    """Concatenate NIfTI volumes along the 4th dimension."""
    imgs = [nib.load(p) for p in paths]
    data = np.concatenate([img.get_fdata() for img in imgs], axis=3)
    merged = nib.Nifti1Image(data, imgs[0].affine, imgs[0].header)
    merged.header["dim"][4] = data.shape[3]
    nib.save(merged, output)


def merge_bval(paths: list[Path], output: Path) -> None:
    """Concatenate bval files (single-row FSL format)."""
    all_vals = []
    for p in paths:
        all_vals.extend(p.read_text().strip().split())
    output.write_text(" ".join(all_vals) + "\n")


def merge_bvec(paths: list[Path], output: Path) -> None:
    """Concatenate bvec files (3-row FSL format)."""
    rows: list[list[str]] = [[], [], []]
    for p in paths:
        lines = p.read_text().strip().splitlines()
        for i, line in enumerate(lines):
            rows[i].extend(line.strip().split())
    output.write_text("\n".join(" ".join(row) for row in rows) + "\n")


def merge_json(paths: list[Path], output: Path) -> None:
    """Copy JSON sidecar from the first split (metadata is identical)."""
    shutil.copy2(paths[0], output)


def strip_nifti_ext(path: Path) -> Path:
    """Remove .nii.gz or .nii extension."""
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7])
    elif name.endswith(".nii"):
        return path.with_name(name[:-4])
    return path


def merge_subject(source_dwi: Path, raw_dwi: Path) -> None:
    """Merge all split groups for one subject."""
    groups = find_split_groups(source_dwi)

    if not groups:
        print(f"  No split DWI files found in {source_dwi}")
        return

    raw_dwi.mkdir(parents=True, exist_ok=True)

    for merged_name, nii_paths in sorted(groups.items()):
        letters = [get_split_letter(p.name) for p in nii_paths]
        print(f"  Merging {len(nii_paths)} splits ({','.join(letters)}) -> {merged_name}")

        out_nii = raw_dwi / merged_name

        # Merge NIfTI
        merge_nifti(nii_paths, out_nii)

        # Merge bval
        bval_paths = [strip_nifti_ext(p).with_suffix(".bval") for p in nii_paths]
        if all(p.exists() for p in bval_paths):
            merge_bval(bval_paths, strip_nifti_ext(out_nii).with_suffix(".bval"))

        # Merge bvec
        bvec_paths = [strip_nifti_ext(p).with_suffix(".bvec") for p in nii_paths]
        if all(p.exists() for p in bvec_paths):
            merge_bvec(bvec_paths, strip_nifti_ext(out_nii).with_suffix(".bvec"))

        # Copy JSON from first split
        json_paths = [strip_nifti_ext(p).with_suffix(".json") for p in nii_paths]
        if json_paths[0].exists():
            merge_json(json_paths, strip_nifti_ext(out_nii).with_suffix(".json"))


def main():
    parser = argparse.ArgumentParser(
        description="Merge split DWI acquisitions (A/B/C) into single files"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/single_pe_rpe/native_res"),
        help="Base directory containing source/ and raw/ (default: data/single_pe_rpe/native_res)",
    )
    args = parser.parse_args()

    source_dir = args.base_dir / "source"
    raw_dir = args.base_dir / "raw"

    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    subjects = sorted([d for d in source_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    print(f"Found {len(subjects)} subject(s)")

    for sub_dir in subjects:
        print(f"\n{sub_dir.name}")
        source_dwi = sub_dir / "dwi"
        raw_dwi = raw_dir / sub_dir.name / "dwi"

        if not source_dwi.is_dir():
            print(f"  No dwi/ folder, skipping")
            continue

        merge_subject(source_dwi, raw_dwi)

    print("\nDone.")


if __name__ == "__main__":
    main()
