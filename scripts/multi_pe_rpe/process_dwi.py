#!/usr/bin/env python3
"""Process DWI data for multi_pe_rpe dataset.

Full pipeline: dwifslpreproc (topup/eddy) → brain masking → tensor fitting
Processes native (1.7mm) and all downsampled resolutions (2.0, 2.5, 3.0, 3.4mm).
Only processes lr/rl phase encoding pair.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add parent directories to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.common.preprocessing import (create_brain_mask, fit_tensors,
                                          register_to_anat_ref,
                                          run_dwifslpreproc)
from utils.cmd_utils import run_command

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
) -> None:
    """Process DWI pair with a specific mode (b0_rpe or full_rpe).

    Parameters
    ----------
    dwi_lr : Path
        LR phase encoding denoised DWI
    dwi_rl : Path
        RL phase encoding denoised DWI
    dwi_dir : Path
        DWI directory (for output)
    mode : str
        Processing mode: "b0_rpe" or "full_rpe"
    nthreads : int
        Number of threads for topup
    anat_ref_image : Path, optional
        Anatomical reference image for registration (e.g., MPRAGE)
    anat_mask_image : Path, optional
        Anatomical brain mask to regrid and use (instead of creating from DWI)
    skip_preproc : bool
        Skip dwifslpreproc, use existing preprocessed DWI
    """
    print(f"  === Mode: {mode} ===")

    if skip_preproc:
        preprocessed, eddy_dir = _expected_preproc_paths(dwi_dir, mode)
        if not preprocessed.exists():
            raise FileNotFoundError(
                f"--skip-preproc: expected preprocessed DWI at {preprocessed}"
            )
        print(f"  Skipping preprocessing, using existing: {preprocessed.name}")
    else:
        # Step 1: Preprocessing (dwifslpreproc)
        print(f"  Step 1: Preprocessing (topup/eddy)...")
        preprocessed, eddy_dir = run_dwifslpreproc(
            dwi_lr, dwi_rl, dwi_dir, mode=mode, nthreads=nthreads
        )
        print(f"    Output: {preprocessed.name}")

    # Step 2: Brain masking
    print(f"  Step 2: Brain masking...")
    prefix = preprocessed.stem.replace(".nii", "")
    final_mask = dwi_dir / f"{prefix}_brain_mask.nii.gz"

    if anat_mask_image is not None and anat_mask_image.exists():
        # Regrid and reorient anatomical mask to match preprocessed DWI exactly
        print(f"    Regridding anatomical mask: {anat_mask_image.name}")

        # Transform mask to match preprocessed DWI template (geometry + strides)
        # Using -template alone doesn't guarantee stride matching, so we also
        # explicitly copy the DWI strides to ensure nibabel loads them identically
        mask_tmp = dwi_dir / f"_tmp_mask_{mode}.nii.gz"
        run_command(
            f"mrtransform {anat_mask_image} -template {preprocessed} "
            f"-interp nearest {mask_tmp} -force",
            verbose=False,
        )

        # Force mask to have identical strides as preprocessed DWI
        run_command(
            f"mrconvert {mask_tmp} -strides {preprocessed} {final_mask} -force",
            verbose=False,
        )

        # Clean up
        mask_tmp.unlink(missing_ok=True)

        print(f"    Brain mask (from anat): {final_mask.name}")
    else:
        # Create brain mask from DWI
        mask_tmp = dwi_dir / f"_mask_tmp_{mode}"
        brain_mask = create_brain_mask(preprocessed, mask_tmp)
        brain_mask.rename(final_mask)
        shutil.rmtree(mask_tmp, ignore_errors=True)
        print(f"    Brain mask (from DWI): {final_mask.name}")

    # Create masked DWI
    masked_dwi = dwi_dir / f"{prefix}_masked.nii.gz"
    run_command(f"mrcalc {preprocessed} {final_mask} -mult {masked_dwi} -force", verbose=False)
    print(f"    Masked DWI: {masked_dwi.name}")

    # Step 3: Tensor fitting
    print(f"  Step 3: Tensor fitting...")
    output_prefix = dwi_dir / prefix
    fit_tensors(preprocessed, output_prefix, final_mask)

    # Step 4: Anatomical registration (optional)
    if anat_ref_image is not None and anat_ref_image.exists():
        print(f"  Step 4: Anatomical registration...")

        dwi_anat, anat_mask = register_to_anat_ref(
            preprocessed_dwi=preprocessed,
            eddy_output_dir=eddy_dir,
            anat_ref_image=anat_ref_image,
            output_dir=dwi_dir,
            anat_mask_image=anat_mask_image,
            dwi_mask=final_mask,
            nthreads=nthreads,
        )

        print(f"  Step 5: Tensor fitting on registered data...")
        anat_output_prefix = dwi_dir / dwi_anat.stem.replace(".nii", "")
        fit_tensors(dwi_anat, anat_output_prefix, anat_mask)

        # Step 6: Tissue segmentation using multi-channel Atropos (FA + S0)
        print(f"  Step 6: Tissue segmentation (ANTs Atropos, multi-channel)...")
        anat_fa = Path(f"{anat_output_prefix}_dti_fa.nii.gz")
        anat_s0 = Path(f"{anat_output_prefix}_dti_s0.nii.gz")

        if anat_fa.exists() and anat_s0.exists():
            seg_path = dwi_dir / f"{anat_output_prefix.name}_segmentation.nii.gz"
            prob_prefix = str(dwi_dir / f"{anat_output_prefix.name}_segmentation_prob")

            # Multi-channel Atropos: combine FA (tissue contrast) + S0 (T2-weighted contrast)
            # Use multiple -a flags for multi-channel input
            cmd = [
                "Atropos",
                "-d", "3",
                "-a", str(anat_fa),              # Channel 1: FA (tissue contrast)
                "-a", str(anat_s0),              # Channel 2: S0 (T2-weighted contrast)
                "-x", str(anat_mask),
                "-i", "KMeans[3]",               # 3-tissue k-means initialization
                "-c", "[5,0]",                   # 5 iterations, no partial volume
                "-m", "[0.1,1x1x1]",             # MRF smoothing (weight=0.1, radius=1)
                "-o", f"[{seg_path},{prob_prefix}_%02d.nii.gz]"
            ]
            run_command(cmd, verbose=False)
            print(f"    Segmentation: {seg_path.name}")

            # Rename probability maps to meaningful names (1=CSF, 2=GM, 3=WM)
            prob_csf = dwi_dir / f"{anat_output_prefix.name}_segmentation_prob_csf.nii.gz"
            prob_gm = dwi_dir / f"{anat_output_prefix.name}_segmentation_prob_gm.nii.gz"
            prob_wm = dwi_dir / f"{anat_output_prefix.name}_segmentation_prob_wm.nii.gz"

            Path(f"{prob_prefix}_01.nii.gz").rename(prob_csf)
            Path(f"{prob_prefix}_02.nii.gz").rename(prob_gm)
            Path(f"{prob_prefix}_03.nii.gz").rename(prob_wm)

            print(f"    Probability maps: {prob_csf.name}, {prob_gm.name}, {prob_wm.name}")
        else:
            print(f"    Skipping segmentation: FA or S0 not found")

    print(f"  ✓ Completed {mode} mode")
    print()


def process_dwi_pair(
    dwi_lr: Path,
    dwi_rl: Path,
    dwi_dir: Path,
    nthreads: int = 0,
    anat_ref_image: Path | None = None,
    anat_mask_image: Path | None = None,
    skip_preproc: bool = False,
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
    """
    print(f"  Processing: {dwi_lr.name}")
    print(f"    Reverse PE: {dwi_rl.name}")
    print()

    # Process with both modes
    process_single_mode(dwi_lr, dwi_rl, dwi_dir, mode="b0_rpe", nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc)
    process_single_mode(dwi_lr, dwi_rl, dwi_dir, mode="full_rpe", nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc)

    print(f"  ✓ Completed all modes for lr/rl pair")


def process_resolution(dwi_dir: Path, resolution_name: str, nthreads: int = 0, skip_preproc: bool = False) -> None:
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
    """
    if not dwi_dir.is_dir():
        print(f"  No dwi/ folder at {dwi_dir}, skipping")
        return

    # Find anatomical reference (1.7mm downsampled MPRAGE)
    # Always use the native res anat/ directory — the 1.7mm MPRAGE is the
    # reference for both native and downsampled DWI resolutions.
    anat_dir = NATIVE_BASE / "anat"
    anat_ref_image = None
    anat_mask_image = None

    if anat_dir.is_dir():
        mprage_file = anat_dir / "mprage_downsampled_1p7.nii.gz"
        if mprage_file.exists():
            anat_ref_image = mprage_file
            print(f"  Found anatomical reference: {anat_ref_image}")
        else:
            print(f"  No mprage_downsampled_1p7.nii.gz found in {anat_dir}")

        # Find anatomical brain mask
        mask_file = anat_dir / "brain_mask_downsampled_1p7.nii.gz"
        if mask_file.exists():
            anat_mask_image = mask_file
            print(f"  Found anatomical mask: {anat_mask_image}")
        else:
            print(f"  No brain_mask_downsampled_1p7.nii.gz found in {anat_dir}")

    # Find lr/rl pair
    pe_pair = find_lr_rl_pair(dwi_dir)

    if not pe_pair:
        print(f"  No lr/rl pair found, skipping")
        return

    dwi_lr, dwi_rl = pe_pair
    print(f"  Found lr/rl pair [{resolution_name}]")

    try:
        process_dwi_pair(dwi_lr, dwi_rl, dwi_dir, nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc)
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
    args = parser.parse_args()

    print("Processing multi_pe_rpe DWI data (lr/rl pair only)")
    print(f"Topup threads: {args.topup_threads}")
    print("=" * 60)
    print()

    # Process native resolution
    native_dwi_dir = NATIVE_BASE / "dwi"
    if native_dwi_dir.is_dir():
        print(f"=== Native resolution (1.7mm) ===")
        process_resolution(native_dwi_dir, "native 1.7mm", nthreads=args.topup_threads, skip_preproc=args.skip_preproc)
    else:
        print(f"Native DWI directory not found: {native_dwi_dir}")
        print()

    # # Process downsampled resolutions
    # for res in DOWNSAMPLED_RESOLUTIONS:
    #     res_str = str(res).replace(".", "p")
    #     dwi_dir = ROOT / f"data/multi_pe_rpe/downsampled_{res_str}mm/processed/dwi"

    #     if dwi_dir.is_dir():
    #         print(f"=== Downsampled resolution ({res}mm) ===")
    #         process_resolution(dwi_dir, f"downsampled {res}mm", nthreads=args.topup_threads, skip_preproc=args.skip_preproc)
    #     else:
    #         print(f"Downsampled DWI directory not found: {dwi_dir}")
    #         print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
