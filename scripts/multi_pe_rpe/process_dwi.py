#!/usr/bin/env python3
"""Process DWI data for multi_pe_rpe dataset.

Full pipeline: dwifslpreproc (topup/eddy) → brain masking → tensor fitting
Processes native (1.7mm) and all downsampled resolutions (2.0, 2.5, 3.0, 3.4mm).
Only processes lr/rl phase encoding pair.
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

NATIVE_BASE = ROOT / "data/multi_pe_rpe/native_res/processed"
DOWNSAMPLED_RESOLUTIONS = [2.0, 2.5, 3.0, 3.4]


def find_lr_rl_pair(dwi_dir: Path) -> tuple[Path, Path] | None:
    """Find lr and rl denoised DWI files.

    Returns
    -------
    tuple[Path, Path] | None
        (lr, rl) tuple or None if not found
    """
    lr_file = dwi_dir / "dwi_lr_denoised.nii.gz"
    rl_file = dwi_dir / "dwi_rl_denoised.nii.gz"

    if lr_file.exists() and rl_file.exists():
        return (lr_file, rl_file)
    return None


def _expected_preproc_paths(dwi_dir: Path, mode: str) -> tuple[Path, Path]:
    """Compute expected preprocessed DWI and eddy output dir paths for a mode."""
    if mode == "b0_rpe":
        output_name = "dwi_lr_preprocessed.nii.gz"
    else:  # full_rpe mode
        output_name = "dwi_merged_preprocessed.nii.gz"
    preprocessed = dwi_dir / output_name
    eddy_dir = dwi_dir / f"{output_name.replace('.nii.gz', '')}_eddy_output"
    return preprocessed, eddy_dir


def process_single_mode(
    dwi_lr: Path,
    dwi_rl: Path,
    dwi_dir: Path,
    mode: str,
    nthreads: int = 0,
    anat_ref_image: Path | None = None,
    anat_mask_image: Path | None = None,
    skip_preproc: bool = False,
    keep_tmp: bool = False,
) -> None:
    """Process DWI pair with a specific mode (b0_rpe or full_rpe).

    Wrapper around common.preprocessing.process_single_mode for multi_pe_rpe dataset.
    """
    # Compute expected paths for skip_preproc case
    preprocessed_path, eddy_dir_path = (
        _expected_preproc_paths(dwi_dir, mode) if skip_preproc else (None, None)
    )

    # Call common processing function
    from scripts.common.preprocessing import process_single_mode as common_process_single_mode
    common_process_single_mode(
        dwi_forward=dwi_lr,
        dwi_reverse=dwi_rl,
        dwi_dir=dwi_dir,
        mode=mode,
        nthreads=nthreads,
        anat_ref_image=anat_ref_image,
        anat_mask_image=anat_mask_image,
        skip_preproc=skip_preproc,
        preprocessed_path=preprocessed_path,
        eddy_dir_path=eddy_dir_path,
        keep_tmp=keep_tmp,
    )


def process_dwi_pair(
    dwi_lr: Path,
    dwi_rl: Path,
    dwi_dir: Path,
    nthreads: int = 0,
    anat_ref_image: Path | None = None,
    anat_mask_image: Path | None = None,
    skip_preproc: bool = False,
    keep_tmp: bool = False,
) -> None:
    """Process lr/rl DWI pair with both b0_rpe and full_rpe modes.

    Parameters
    ----------
    dwi_lr : Path
        LR phase encoding denoised DWI
    dwi_rl : Path
        RL phase encoding denoised DWI
    dwi_dir : Path
        DWI directory (for output)
    nthreads : int
        Number of threads for topup
    anat_ref_image : Path, optional
        Anatomical reference image for registration
    anat_mask_image : Path, optional
        Anatomical brain mask to regrid and use
    skip_preproc : bool
        Skip dwifslpreproc, use existing preprocessed DWI
    keep_tmp : bool
        Keep temporary directory (contains scratch dir with eddy QC outputs)
    """
    print(f"  Processing: {dwi_lr.name}")
    print(f"    Reverse PE: {dwi_rl.name}")
    print()

    # Process with both modes
    process_single_mode(dwi_lr, dwi_rl, dwi_dir, mode="b0_rpe", nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc, keep_tmp=keep_tmp)
    process_single_mode(dwi_lr, dwi_rl, dwi_dir, mode="full_rpe", nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc, keep_tmp=keep_tmp)

    print(f"  ✓ Completed all modes for lr/rl pair")


def process_resolution(dwi_dir: Path, resolution_name: str, nthreads: int = 0, skip_preproc: bool = False, keep_tmp: bool = False) -> None:
    """Process DWI data at one resolution.

    Parameters
    ----------
    dwi_dir : Path
        DWI directory (e.g., native_res/processed/dwi or downsampled_2p0mm/processed/dwi)
    resolution_name : str
        Resolution identifier (for logging)
    nthreads : int
        Number of threads for topup
    skip_preproc : bool
        Skip dwifslpreproc, use existing preprocessed DWI
    keep_tmp : bool
        Keep temporary directory (contains scratch dir with eddy QC outputs)
    """
    if not dwi_dir.is_dir():
        print(f"  No dwi/ folder at {dwi_dir}, skipping")
        return

    # No anatomical reference for multi_pe_rpe dataset
    anat_ref_image = None
    anat_mask_image = None

    # Find lr/rl pair
    pe_pair = find_lr_rl_pair(dwi_dir)

    if not pe_pair:
        print(f"  No lr/rl pair found, skipping")
        return

    dwi_lr, dwi_rl = pe_pair
    print(f"  Found lr/rl pair [{resolution_name}]")

    try:
        process_dwi_pair(dwi_lr, dwi_rl, dwi_dir, nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc, keep_tmp=keep_tmp)
    except Exception as e:
        print(f"  ERROR: Failed to process lr/rl pair: {e}")
        return

    print()


def main():
    """Process DWI data at all resolutions."""
    parser = argparse.ArgumentParser(
        description="Process DWI data for multi_pe_rpe dataset"
    )
    parser.add_argument(
        "--topup-threads",
        type=int,
        default=0,
        help="Number of threads for topup (default: 0=all available)",
    )
    parser.add_argument(
        "--skip-preproc",
        action="store_true",
        help="Skip dwifslpreproc, use existing preprocessed DWI in output dirs",
    )
    parser.add_argument(
        "--keep-tmp",
        default=True,
        type=lambda x: x.lower() not in ['false', '0', 'no'],
        help="Keep temporary directory (contains scratch dir with eddy QC outputs) [default: True]",
    )
    args = parser.parse_args()

    print("Processing multi_pe_rpe DWI data (lr/rl pair only)")
    print(f"Topup threads: {args.topup_threads}")
    print(f"Keep tmp: {args.keep_tmp}")
    print("=" * 60)
    print()

    # Process native resolution
    native_dwi_dir = NATIVE_BASE / "dwi"
    if native_dwi_dir.is_dir():
        print(f"=== Native resolution (1.7mm) ===")
        process_resolution(native_dwi_dir, "native 1.7mm", nthreads=args.topup_threads, skip_preproc=args.skip_preproc, keep_tmp=args.keep_tmp)
    else:
        print(f"Native DWI directory not found: {native_dwi_dir}")
        print()

    # Process downsampled resolutions
    for res in DOWNSAMPLED_RESOLUTIONS:
        res_str = str(res).replace(".", "p")
        dwi_dir = ROOT / f"data/multi_pe_rpe/downsampled_{res_str}mm/processed/dwi"

        if dwi_dir.is_dir():
            print(f"=== Downsampled resolution ({res}mm) ===")
            process_resolution(dwi_dir, f"downsampled {res}mm", nthreads=args.topup_threads, skip_preproc=args.skip_preproc, keep_tmp=args.keep_tmp)
        else:
            print(f"Downsampled DWI directory not found: {dwi_dir}")
            print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
