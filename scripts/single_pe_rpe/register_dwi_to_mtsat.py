#!/usr/bin/env python3
"""Register DWI to MTsat and apply transform to WM segmentation.

For each subject, resolution, and mode (b0_rpe, full_rpe):
  1. Register DWI space → MTsat using mean b=0 + DTI FA (two-stage rigid)
  2. Apply transform to c2 (WM) SPM segmentation map

Input files (per subject dwi_dir / prefix):
  {prefix}_mean_b0.nii.gz        — registration moving channel 1
  {prefix}_dti_fa.nii.gz         — registration moving channel 2
  {prefix}_brain_mask.nii.gz     — DWI brain mask
  c2{prefix}_mean_b0.nii         — SPM WM probability map (uncompressed)

Fixed images (per subject, in native_res mpm/):
  {subject}_MTsat.nii (or .nii.gz) — native resolution
  mtsat_brain_mask.nii.gz

Outputs (per subject dwi_dir / prefix):
  {prefix}_reg/                  — ANTs transforms
  {prefix}_wm_in_anat.nii.gz     — WM map warped to MTsat space
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.common.preprocessing import register_to_anat_ref_rigid
from utils.cmd_utils import run_command

NATIVE_BASE = ROOT / "data/single_pe_rpe/native_res/processed"
DOWNSAMPLED_BASE = ROOT / "data/single_pe_rpe/downsampled_2p5mm/processed"


def find_dwi_files(dwi_dir: Path, mode: str) -> list[tuple[str, Path, Path, Path, Path]]:
    """Find DWI files for a given mode.

    Returns
    -------
    list[tuple[str, Path, Path, Path, Path]]
        List of (prefix, mean_b0, fa, brain_mask, c2_wm) tuples
    """
    files = []

    if mode == "b0_rpe":
        # Pattern: *_dir-*_preprocessed
        pattern = "*_dir-*_preprocessed_mean_b0.nii.gz"
    else:  # full_rpe
        # Pattern: *_merged_preprocessed
        pattern = "*_merged_preprocessed_mean_b0.nii.gz"

    mean_b0_files = sorted(dwi_dir.glob(pattern))

    for mean_b0 in mean_b0_files:
        # Extract prefix
        prefix = mean_b0.name.replace("_mean_b0.nii.gz", "")

        # Construct other file paths
        fa = dwi_dir / f"{prefix}_dti_fa.nii.gz"
        brain_mask = dwi_dir / f"{prefix}_brain_mask.nii.gz"
        c2_wm = dwi_dir / f"c2{prefix}_mean_b0.nii"

        # Check if all files exist
        if fa.exists() and brain_mask.exists() and c2_wm.exists():
            files.append((prefix, mean_b0, fa, brain_mask, c2_wm))

    return files


def register_subject_mode(
    subject_dir: Path,
    mode: str,
    resolution_name: str,
    nthreads: int,
) -> None:
    """Register DWI to MTsat for one subject and mode."""
    dwi_dir = subject_dir / "dwi"

    if not dwi_dir.is_dir():
        print(f"    No dwi/ directory, skipping")
        return

    # Find anatomical reference (always in native_res)
    subject_id = subject_dir.name
    native_mpm_dir = NATIVE_BASE / subject_id / "mpm"

    # Find MTsat native file (MPMs are always at native resolution)
    mtsat_files = sorted(native_mpm_dir.glob(f"{subject_id}_MTsat.nii*"))
    if not mtsat_files:
        print(f"    MTsat not found, skipping")
        return
    mtsat_ref = mtsat_files[0]

    # Find brain mask
    brain_mask_ref = native_mpm_dir / "mtsat_brain_mask.nii.gz"
    if not brain_mask_ref.exists():
        print(f"    MTsat brain mask not found, skipping")
        return

    # Find DWI files for this mode
    dwi_files = find_dwi_files(dwi_dir, mode)

    if not dwi_files:
        print(f"    No DWI files found for mode {mode}, skipping")
        return

    print(f"    Mode: {mode} ({len(dwi_files)} file(s))")

    for prefix, mean_b0, fa, brain_mask, c2_wm in dwi_files:
        if len(dwi_files) > 1:
            print(f"      Processing: {prefix}")

        # Register DWI to MTsat
        reg_dir = dwi_dir / f"{prefix}_reg"
        try:
            transforms = register_to_anat_ref_rigid(
                mean_b0=mean_b0,
                fa=fa,
                anat_ref=mtsat_ref,
                anat_mask=brain_mask_ref,
                dwi_mask=brain_mask,
                output_dir=reg_dir,
                nthreads=nthreads,
            )
        except Exception as e:
            print(f"      ERROR: Registration failed: {e}")
            continue

        # Apply transform to WM segmentation
        wm_in_anat = dwi_dir / f"{prefix}_wm_in_anat.nii.gz"
        try:
            run_command(
                [
                    "antsApplyTransforms",
                    "-d", "3",
                    "-i", str(c2_wm),
                    "-r", str(mtsat_ref),
                    "-o", str(wm_in_anat),
                    "-t", str(transforms[0]),  # affine
                    "--interpolation", "Linear",
                ],
                verbose=False,
            )
            if len(dwi_files) > 1:
                print(f"      WM in anat: {wm_in_anat.name}")
        except Exception as e:
            print(f"      ERROR: Transform application failed: {e}")


def process_subject(
    subject_dir: Path,
    resolution_name: str,
    nthreads: int,
) -> None:
    """Process one subject at one resolution."""
    subject_id = subject_dir.name
    print(f"  === {subject_id} [{resolution_name}] ===")

    # Process both modes
    for mode in ["b0_rpe", "full_rpe"]:
        try:
            register_subject_mode(subject_dir, mode, resolution_name, nthreads)
        except Exception as e:
            print(f"    ERROR ({mode}): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Register DWI to MTsat and warp WM segmentation"
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=16,
        help="Number of ANTs threads (default: 16)",
    )
    args = parser.parse_args()

    print("Registering DWI → MTsat for all subjects and resolutions")
    print(f"Threads: {args.nthreads}")
    print("=" * 60)

    # Process native resolution
    if NATIVE_BASE.is_dir():
        subjects = sorted([d for d in NATIVE_BASE.iterdir()
                          if d.is_dir() and d.name.startswith("sub-")])

        print(f"\n=== Native resolution (1.6mm): {len(subjects)} subject(s) ===")
        for subject_dir in subjects:
            try:
                process_subject(subject_dir, "native 1.6mm", args.nthreads)
            except Exception as e:
                print(f"  ERROR ({subject_dir.name}): {e}")
            print()

    # Process downsampled resolution
    if DOWNSAMPLED_BASE.is_dir():
        subjects = sorted([d for d in DOWNSAMPLED_BASE.iterdir()
                          if d.is_dir() and d.name.startswith("sub-")])

        print(f"\n=== Downsampled resolution (2.5mm): {len(subjects)} subject(s) ===")
        for subject_dir in subjects:
            try:
                process_subject(subject_dir, "downsampled 2.5mm", args.nthreads)
            except Exception as e:
                print(f"  ERROR ({subject_dir.name}): {e}")
            print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
