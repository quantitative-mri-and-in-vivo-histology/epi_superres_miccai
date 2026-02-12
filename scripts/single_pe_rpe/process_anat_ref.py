#!/usr/bin/env python3
"""Process anatomical reference for single_pe_rpe dataset.

Uses MTsat map for brain extraction and tissue segmentation, then downsamples
all MPM maps along with brain mask and segmentation to 1.6mm isotropic.
"""

import sys
from pathlib import Path

# Add utils to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cmd_utils import run_command


INPUT_BASE = ROOT / "data/single_pe_rpe/native_res/processed"
TARGET_VOXEL = 1.6


def brain_extract_and_segment(mtsat_path: Path, output_dir: Path) -> dict[str, Path]:
    """Brain extraction and tissue segmentation using MTsat map.

    Pipeline:
    1. Brain extraction (ANTs)
    2. Tissue segmentation (ANTs Atropos): 1=CSF, 2=GM, 3=WM

    Parameters
    ----------
    mtsat_path : Path
        Input MTsat NIfTI file.
    output_dir : Path
        Output directory for brain mask and segmentation.

    Returns
    -------
    dict[str, Path]
        Dictionary with paths to brain_mask and segmentation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Brain extraction with ANTs using MNI template
    print("  Brain extraction (ANTs)...")
    brain_path = output_dir / "mtsat_brain.nii.gz"
    brain_mask_path = output_dir / "brain_mask.nii.gz"

    template_path = ROOT / "data/templates/MNI152_T1_1mm.nii.gz"
    template_mask_path = ROOT / "data/templates/MNI152_T1_1mm_brain_mask.nii.gz"
    brain_extraction_prefix = output_dir / "ants_brain_extraction_"

    cmd = [
        "antsBrainExtraction.sh",
        "-d", "3",
        "-a", str(mtsat_path),
        "-e", str(template_path),
        "-m", str(template_mask_path),
        "-o", str(brain_extraction_prefix)
    ]
    run_command(cmd, verbose=False)

    # antsBrainExtraction.sh outputs:
    # - {prefix}BrainExtractionMask.nii.gz (brain mask)
    # - {prefix}BrainExtractionBrain.nii.gz (brain extracted image)
    ants_mask = Path(f"{brain_extraction_prefix}BrainExtractionMask.nii.gz")
    ants_brain = Path(f"{brain_extraction_prefix}BrainExtractionBrain.nii.gz")

    if ants_mask.exists():
        ants_mask.rename(brain_mask_path)
    if ants_brain.exists():
        ants_brain.rename(brain_path)

    print(f"    Saved brain mask: {brain_mask_path.name}")

    # Tissue segmentation with ANTs Atropos (1=CSF, 2=GM, 3=WM)
    print("  Tissue segmentation (ANTs Atropos)...")
    seg_path = output_dir / "segmentation.nii.gz"
    prob_prefix = str(output_dir / "segmentation_prob")

    cmd = [
        "Atropos",
        "-d", "3",
        "-a", str(brain_path),
        "-x", str(brain_mask_path),
        "-i", "KMeans[3]",        # 3-tissue k-means initialization
        "-c", "[5,0]",             # 5 iterations, no partial volume
        "-m", "[0.1,1x1x1]",       # MRF smoothing (weight=0.1, radius=1)
        "-o", f"[{seg_path},{prob_prefix}_%02d.nii.gz]"
    ]
    run_command(cmd, verbose=False)
    print(f"    Saved segmentation: {seg_path.name}")

    # Rename probability maps to meaningful names (1=CSF, 2=GM, 3=WM)
    prob_csf = output_dir / "segmentation_prob_csf.nii.gz"
    prob_gm = output_dir / "segmentation_prob_gm.nii.gz"
    prob_wm = output_dir / "segmentation_prob_wm.nii.gz"

    Path(f"{prob_prefix}_01.nii.gz").rename(prob_csf)
    Path(f"{prob_prefix}_02.nii.gz").rename(prob_gm)
    Path(f"{prob_prefix}_03.nii.gz").rename(prob_wm)

    print(f"    Saved probability maps: {prob_csf.name}, {prob_gm.name}, {prob_wm.name}")

    # Clean up intermediate brain-extracted image
    if brain_path.exists():
        brain_path.unlink()

    return {
        "brain_mask": brain_mask_path,
        "segmentation": seg_path,
        "prob_csf": prob_csf,
        "prob_gm": prob_gm,
        "prob_wm": prob_wm,
    }


def downsample_image(input_path: Path, output_path: Path, voxel: float, interp: str = "sinc"):
    """Downsample image using MRtrix mrgrid.

    Parameters
    ----------
    input_path : Path
        Input image file.
    output_path : Path
        Output downsampled file.
    voxel : float
        Target isotropic voxel size in mm.
    interp : str
        Interpolation method ('sinc' or 'nearest').
    """
    cmd = [
        "mrgrid", str(input_path), "regrid", str(output_path),
        "-voxel", str(voxel), "-interp", interp, "-force"
    ]
    run_command(cmd, verbose=False)


def process_subject(subject_dir: Path, target_voxel: float):
    """Process one subject: brain extraction, segmentation, and downsampling.

    Parameters
    ----------
    subject_dir : Path
        Subject directory in processed/ (e.g., processed/sub-V06460).
    target_voxel : float
        Target isotropic voxel size in mm.
    """
    mpm_dir = subject_dir / "mpm"

    if not mpm_dir.is_dir():
        print("  No mpm/ folder, skipping")
        return

    # Find MTsat map for brain extraction
    mtsat_path = mpm_dir / f"{subject_dir.name}_MTsat.nii"
    if not mtsat_path.exists():
        print("  No MTsat map found, skipping")
        return

    # Find all MPM maps (exclude already downsampled)
    mpm_files = sorted([f for f in mpm_dir.glob("*.nii")
                       if "_downsampled" not in f.name])

    if not mpm_files:
        print("  No MPM maps found, skipping")
        return

    print(f"  Found {len(mpm_files)} MPM map(s)")

    # Brain extraction and segmentation using MTsat
    anat_outputs = brain_extract_and_segment(mtsat_path, mpm_dir)

    # Downsample all MPM maps
    print("  Downsampling MPM maps...")
    for mpm_file in mpm_files:
        output_name = f"{mpm_file.stem}_downsampled.nii.gz"
        output_file = mpm_dir / output_name
        print(f"    {mpm_file.name} -> {output_name}")
        downsample_image(mpm_file, output_file, target_voxel, interp="sinc")

    # Downsample brain mask and segmentation
    print("  Downsampling brain mask and segmentation...")
    for key, src_path in anat_outputs.items():
        stem = src_path.stem.replace('.nii', '')  # Handle .nii.gz
        suffix = f"_downsampled_{str(target_voxel).replace('.', 'p')}"
        dst_name = f"{stem}{suffix}.nii.gz"
        dst_path = mpm_dir / dst_name

        # Use nearest-neighbor for masks and segmentations (label maps)
        interp = "nearest" if ("mask" in key or "segmentation" in key) else "sinc"
        print(f"    {src_path.name} ({interp}) -> {dst_name}")
        downsample_image(src_path, dst_path, target_voxel, interp=interp)

    print(f"  Saved all outputs to: {mpm_dir}")


def main():
    """Process all subjects in the single_pe_rpe dataset."""
    if not INPUT_BASE.is_dir():
        raise ValueError(f"Input directory does not exist: {INPUT_BASE}")

    subjects = sorted([d for d in INPUT_BASE.iterdir()
                      if d.is_dir() and d.name.startswith("sub-")])

    print(f"Processing single_pe_rpe anatomical reference")
    print(f"Target resolution: {TARGET_VOXEL}mm isotropic")
    print(f"Found {len(subjects)} subject(s)")
    print()

    for subject_dir in subjects:
        print(f"=== {subject_dir.name} ===")

        try:
            process_subject(subject_dir, TARGET_VOXEL)
        except Exception as e:
            print(f"  ERROR: Failed to process {subject_dir.name}: {e}")
            continue

        print()

    print("Done.")


if __name__ == "__main__":
    main()
