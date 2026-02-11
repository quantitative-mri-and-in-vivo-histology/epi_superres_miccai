#!/usr/bin/env python3
"""Process DWI data for single_pe_rpe dataset.

Full pipeline: dwifslpreproc (topup/eddy) → brain masking → tensor fitting
Processes both native (1.6mm) and downsampled (2.5mm) resolutions.
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

NATIVE_BASE = ROOT / "data/single_pe_rpe/native_res/processed"
DOWNSAMPLED_BASE = ROOT / "data/single_pe_rpe/downsampled_2p5mm/processed"


def find_pe_pairs(dwi_dir: Path) -> list[tuple[Path, Path]]:
    """Find pairs of denoised DWI files with opposite PE directions.

    Returns
    -------
    list[tuple[Path, Path]]
        List of (forward_pe, reverse_pe) tuples
    """
    denoised_files = sorted(dwi_dir.glob("*_denoised.nii.gz"))

    # Group by base name (everything except dir-XX)
    from collections import defaultdict
    groups = defaultdict(list)

    for f in denoised_files:
        # Extract PE direction from filename (e.g., dir-AP, dir-PA)
        name = f.name
        if "_dir-" in name:
            # Split on _dir- and take everything before it
            base = name.split("_dir-")[0]
            groups[base].append(f)

    # Pair up AP/PA (or other opposite PE directions)
    pairs = []
    for base, files in groups.items():
        if len(files) == 2:
            # Assume first is forward, second is reverse (alphabetically AP < PA)
            pairs.append((files[0], files[1]))

    return pairs


def _expected_preproc_paths(dwi: Path, dwi_dir: Path, mode: str) -> tuple[Path, Path]:
    """Compute expected preprocessed DWI and eddy output dir paths for a mode."""
    if mode == "b0_rpe":
        output_name = dwi.name.replace("_denoised", "_preprocessed")
    else:
        base_name = dwi.name.split("_dir-")[0]
        output_name = f"{base_name}_merged_preprocessed.nii.gz"
    preprocessed = dwi_dir / output_name
    eddy_dir = dwi_dir / f"{output_name.replace('.nii.gz', '')}_eddy_output"
    return preprocessed, eddy_dir


def process_single_mode(
    dwi: Path,
    dwi_rpe: Path,
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
    dwi : Path
        Forward PE denoised DWI
    dwi_rpe : Path
        Reverse PE denoised DWI
    dwi_dir : Path
        DWI directory (for output)
    mode : str
        Processing mode: "b0_rpe" or "full_rpe"
    nthreads : int
        Number of threads for topup
    anat_ref_image : Path, optional
        Anatomical reference image for registration (e.g., MTsat)
    anat_mask_image : Path, optional
        Anatomical brain mask to regrid and use (instead of creating from DWI)
    skip_preproc : bool
        Skip dwifslpreproc, use existing preprocessed DWI
    """
    print(f"  === Mode: {mode} ===")

    if skip_preproc:
        preprocessed, eddy_dir = _expected_preproc_paths(dwi, dwi_dir, mode)
        if not preprocessed.exists():
            raise FileNotFoundError(
                f"--skip-preproc: expected preprocessed DWI at {preprocessed}"
            )
        print(f"  Skipping preprocessing, using existing: {preprocessed.name}")
    else:
        # Step 1: Preprocessing (dwifslpreproc)
        print(f"  Step 1: Preprocessing (topup/eddy)...")
        preprocessed, eddy_dir = run_dwifslpreproc(
            dwi, dwi_rpe, dwi_dir, mode=mode, nthreads=nthreads
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
        else:
            print(f"    Skipping segmentation: FA or S0 not found")

    print(f"  ✓ Completed {mode} mode")
    print()


def process_dwi_pair(
    dwi: Path,
    dwi_rpe: Path,
    dwi_dir: Path,
    nthreads: int = 0,
    anat_ref_image: Path | None = None,
    anat_mask_image: Path | None = None,
    skip_preproc: bool = False,
) -> None:
    """Process one pair of DWI files with both b0_rpe and full_rpe modes.

    Parameters
    ----------
    dwi : Path
        Forward PE denoised DWI
    dwi_rpe : Path
        Reverse PE denoised DWI
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
    print(f"  Processing: {dwi.name}")
    print(f"    Reverse PE: {dwi_rpe.name}")
    print()

    # Process with both mode
    process_single_mode(dwi, dwi_rpe, dwi_dir, mode="b0_rpe", nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc)
    process_single_mode(dwi, dwi_rpe, dwi_dir, mode="full_rpe", nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc)

    print(f"  ✓ Completed all modes for: {dwi.name}")


def process_subject(subject_dir: Path, resolution_name: str, nthreads: int = 0, skip_preproc: bool = False) -> None:
    """Process all DWI pairs for one subject at one resolution.

    Parameters
    ----------
    subject_dir : Path
        Subject directory (e.g., processed/sub-V06460)
    resolution_name : str
        Resolution identifier (for logging)
    nthreads : int
        Number of threads for topup
    skip_preproc : bool
        Skip dwifslpreproc, use existing preprocessed DWI
    """
    subject_id = subject_dir.name
    dwi_dir = subject_dir / "dwi"

    if not dwi_dir.is_dir():
        print(f"  No dwi/ folder, skipping")
        return

    # Find anatomical reference (1.6mm downsampled MTsat)
    # Always use the native res mpm/ directory — the 1.6mm MTsat is the
    # reference for both native and downsampled DWI resolutions.
    anat_ref_image = None
    anat_mask_image = None
    mpm_dir = subject_dir / "mpm"
    if not mpm_dir.is_dir():
        # Fall back to native res mpm/ for downsampled subjects
        mpm_dir = NATIVE_BASE / subject_id / "mpm"

    if mpm_dir.is_dir():
        mtsat_files = list(mpm_dir.glob(f"{subject_id}_MTsat_downsampled.nii.gz"))
        if mtsat_files:
            anat_ref_image = mtsat_files[0]
            print(f"  Found anatomical reference: {anat_ref_image}")
        else:
            print(f"  No MTsat_downsampled found in {mpm_dir}")

        # Find anatomical brain mask
        mask_files = list(mpm_dir.glob("brain_mask.nii.gz"))
        if mask_files:
            anat_mask_image = mask_files[0]
            print(f"  Found anatomical mask: {anat_mask_image}")
        else:
            print(f"  No brain_mask.nii.gz found in {mpm_dir}")

    # Find PE pairs
    pe_pairs = find_pe_pairs(dwi_dir)

    if not pe_pairs:
        print(f"  No PE pairs found, skipping")
        return

    print(f"  Found {len(pe_pairs)} PE pair(s) [{resolution_name}]")

    for dwi, dwi_rpe in pe_pairs:
        try:
            process_dwi_pair(dwi, dwi_rpe, dwi_dir, nthreads=nthreads, anat_ref_image=anat_ref_image, anat_mask_image=anat_mask_image, skip_preproc=skip_preproc)
        except Exception as e:
            print(f"  ERROR: Failed to process {dwi.name}: {e}")
            continue

        print()


def main():
    """Process all subjects at both resolutions."""
    parser = argparse.ArgumentParser(
        description="Process DWI data for single_pe_rpe dataset"
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

    print("Processing single_pe_rpe DWI data")
    print(f"Topup threads: {args.topup_threads}")
    print("=" * 60)
    print()

    # Process native resolution
    if NATIVE_BASE.is_dir():
        subjects = sorted([d for d in NATIVE_BASE.iterdir()
                          if d.is_dir() and d.name.startswith("sub-")])

        print(f"Native resolution (1.6mm): {len(subjects)} subject(s)")
        print()

        for subject_dir in subjects:
            print(f"=== {subject_dir.name} [native 1.6mm] ===")
            try:
                process_subject(subject_dir, "native 1.6mm", nthreads=args.topup_threads, skip_preproc=args.skip_preproc)
            except Exception as e:
                print(f"  ERROR: {e}")
            print()

    # Process downsampled resolution
    if DOWNSAMPLED_BASE.is_dir():
        subjects = sorted([d for d in DOWNSAMPLED_BASE.iterdir()
                          if d.is_dir() and d.name.startswith("sub-")])

        print(f"Downsampled resolution (2.5mm): {len(subjects)} subject(s)")
        print()

        for subject_dir in subjects:
            print(f"=== {subject_dir.name} [downsampled 2.5mm] ===")
            try:
                process_subject(subject_dir, "downsampled 2.5mm", nthreads=args.topup_threads, skip_preproc=args.skip_preproc)
            except Exception as e:
                print(f"  ERROR: {e}")
            print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
