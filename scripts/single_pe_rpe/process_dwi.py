#!/usr/bin/env python3
"""Process DWI data for single_pe_rpe dataset.

Full pipeline: dwifslpreproc (topup/eddy) → brain masking → tensor fitting
Processes both native (1.6mm) and downsampled (2.5mm) resolutions.
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path for imports
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

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
    else:  # full_rpe mode
        if "_dir-" in dwi.name:
            # Extract everything before PE direction
            base_name = dwi.name.split("_dir-")[0]
        else:
            # No PE direction in filename, remove _denoised extension
            base_name = dwi.name.replace("_denoised.nii.gz", "").replace("_denoised.nii", "")
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

    Wrapper around common.preprocessing.process_single_mode for single_pe_rpe dataset.
    """
    # Compute expected paths for skip_preproc case
    preprocessed_path, eddy_dir_path = (
        _expected_preproc_paths(dwi, dwi_dir, mode) if skip_preproc else (None, None)
    )

    # Call common processing function
    from scripts.common.preprocessing import process_single_mode as common_process_single_mode
    common_process_single_mode(
        dwi_forward=dwi,
        dwi_reverse=dwi_rpe,
        dwi_dir=dwi_dir,
        mode=mode,
        nthreads=nthreads,
        anat_ref_image=anat_ref_image,
        anat_mask_image=anat_mask_image,
        skip_preproc=skip_preproc,
        preprocessed_path=preprocessed_path,
        eddy_dir_path=eddy_dir_path,
    )


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
