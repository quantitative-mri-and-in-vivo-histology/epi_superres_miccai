#!/usr/bin/env python3
"""Register DWI to MPRAGE and apply transform to WM segmentation.

For each resolution and mode (b0_rpe, full_rpe):
  1. Register DWI space → MPRAGE using mean b=0 + DTI FA (two-stage rigid)
  2. Apply transform to c2 (WM) SPM segmentation map

Input files (per dwi_dir / prefix):
  {prefix}_mean_b0.nii.gz        — registration moving channel 1
  {prefix}_dti_fa.nii.gz         — registration moving channel 2
  {prefix}_brain_mask.nii.gz     — DWI brain mask (SyN stage)
  c2{prefix}_mean_b0.nii         — SPM WM probability map (uncompressed)

Fixed images (shared across all resolutions):
  anat/mprage.nii.gz
  anat/mprage_brain_mask.nii.gz

Outputs (per dwi_dir / prefix):
  {prefix}_reg/                  — ANTs transforms
  {prefix}_wm_in_anat.nii.gz     — WM map warped to MPRAGE space
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.common.preprocessing import register_to_anat_ref_rigid
from utils.cmd_utils import run_command

NATIVE_BASE  = ROOT / "data/multi_pe_rpe/native_res/processed"
ANAT_DIR     = NATIVE_BASE / "anat"
MPRAGE       = ANAT_DIR / "mprage.nii.gz"
MPRAGE_MASK  = ANAT_DIR / "mprage_brain_mask.nii.gz"

DOWNSAMPLED_RESOLUTIONS = [2.0, 2.5, 3.0, 3.4]

MODES = {
    "b0_rpe":   "dwi_lr_preprocessed",
    "full_rpe": "dwi_merged_preprocessed",
}


def register_mode(dwi_dir: Path, prefix: str, nthreads: int) -> None:
    mean_b0    = dwi_dir / f"{prefix}_mean_b0.nii.gz"
    fa         = dwi_dir / f"{prefix}_dti_fa.nii.gz"
    brain_mask = dwi_dir / f"{prefix}_brain_mask.nii.gz"
    c2_wm      = dwi_dir / f"c2{prefix}_mean_b0.nii"

    for path, label in [
        (mean_b0,    "mean_b0"),
        (fa,         "dti_fa"),
        (brain_mask, "brain_mask"),
        (c2_wm,      "c2 WM segmentation"),
    ]:
        if not path.exists():
            print(f"    Skipping: {label} not found ({path.name})")
            return

    reg_dir = dwi_dir / f"{prefix}_reg"
    transforms = register_to_anat_ref_rigid(
        mean_b0=mean_b0,
        fa=fa,
        anat_ref=MPRAGE,
        anat_mask=MPRAGE_MASK,
        dwi_mask=brain_mask,
        output_dir=reg_dir,
        nthreads=nthreads,
    )

    wm_in_anat = dwi_dir / f"{prefix}_wm_in_anat.nii.gz"
    print(f"    Applying transform to WM segmentation...")
    run_command(
        [
            "antsApplyTransforms",
            "-d", "3",
            "-i", str(c2_wm),
            "-r", str(MPRAGE),
            "-o", str(wm_in_anat),
            "-t", str(transforms[0]),   # affine
            "--interpolation", "Linear",
        ],
        verbose=False,
    )
    print(f"    WM in anat: {wm_in_anat.name}")


def register_resolution(dwi_dir: Path, res_name: str, nthreads: int) -> None:
    if not dwi_dir.is_dir():
        print(f"  Skipping {res_name}: directory not found")
        return

    print(f"\n=== {res_name} ===")
    for mode, prefix in MODES.items():
        print(f"  Mode: {mode}")
        try:
            register_mode(dwi_dir, prefix, nthreads)
        except Exception as e:
            print(f"  ERROR ({mode}): {e}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Register DWI to MPRAGE and warp WM segmentation"
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=16,
        help="Number of ANTs threads (default: 16)",
    )
    args = parser.parse_args()

    for path, label in [(MPRAGE, "MPRAGE"), (MPRAGE_MASK, "MPRAGE brain mask")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print("Registering DWI → MPRAGE for all resolutions and modes")
    print(f"Fixed: {MPRAGE.name}")
    print(f"Threads: {args.nthreads}")
    print("=" * 60)

    native_dwi_dir = NATIVE_BASE / "dwi"
    register_resolution(native_dwi_dir, "native 1.7mm", args.nthreads)

    for res in DOWNSAMPLED_RESOLUTIONS:
        res_str = str(res).replace(".", "p")
        dwi_dir = ROOT / f"data/multi_pe_rpe/downsampled_{res_str}mm/processed/dwi"
        register_resolution(dwi_dir, f"downsampled {res}mm", args.nthreads)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
