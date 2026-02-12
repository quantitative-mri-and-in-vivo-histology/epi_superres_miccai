#!/usr/bin/env python3
"""Process anatomical reference for multi_pe_rpe dataset.

Uses MPRAGE (T1w) for brain extraction and tissue segmentation, with
denoising and N4 bias field correction, then downsamples to 1.7mm isotropic.
"""

import sys
from pathlib import Path

# Add utils to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cmd_utils import run_command

INPUT_IMAGE = ROOT / "data/multi_pe_rpe/native_res/raw/anat/mprage.nii.gz"
OUTPUT_DIR = ROOT / "data/multi_pe_rpe/native_res/processed/anat"
TARGET_VOXEL = 1.7


def brain_extract_and_segment(mprage_path: Path, output_dir: Path) -> dict[str, Path]:
    """Brain extraction and tissue segmentation using MPRAGE.

    Pipeline:
    1. Denoise (ANTs DenoiseImage)
    2. N4 bias field correction (ANTs N4BiasFieldCorrection)
    3. Brain extraction (FSL BET)
    4. Tissue segmentation (ANTs Atropos): 1=CSF, 2=GM, 3=WM

    Parameters
    ----------
    mprage_path : Path
        Input MPRAGE NIfTI file.
    output_dir : Path
        Output directory for processed images, brain mask, and segmentation.

    Returns
    -------
    dict[str, Path]
        Dictionary with paths to mprage, brain_mask, and segmentation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Denoise with ANTs DenoiseImage
    print("  Denoising (ANTs DenoiseImage)...")
    denoised_path = output_dir / "mprage_denoised.nii.gz"
    cmd = ["DenoiseImage", "-d", "3", "-i", str(mprage_path), "-o", str(denoised_path)]
    run_command(cmd, verbose=False)

    # N4 bias field correction
    print("  N4 bias field correction...")
    mprage_corrected_path = output_dir / "mprage.nii.gz"
    cmd = ["N4BiasFieldCorrection", "-d", "3", "-i", str(denoised_path), "-o", str(mprage_corrected_path)]
    run_command(cmd, verbose=False)
    print(f"    Saved corrected MPRAGE: {mprage_corrected_path.name}")

    # Clean up intermediate denoised image
    if denoised_path.exists():
        denoised_path.unlink()

    # Brain extraction with ANTs using MNI template
    print("  Brain extraction (ANTs)...")
    brain_path = output_dir / "mprage_brain.nii.gz"
    brain_mask_path = output_dir / "brain_mask.nii.gz"

    template_path = ROOT / "data/templates/MNI152_T1_1mm.nii.gz"
    template_mask_path = ROOT / "data/templates/MNI152_T1_1mm_brain_mask.nii.gz"
    brain_extraction_prefix = output_dir / "ants_brain_extraction_"

    cmd = [
        "antsBrainExtraction.sh",
        "-d", "3",
        "-a", str(mprage_corrected_path),
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
        "mprage": mprage_corrected_path,
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


def main():
    """Process MPRAGE anatomical reference for multi_pe_rpe dataset."""
    if not INPUT_IMAGE.exists():
        raise ValueError(f"Input image does not exist: {INPUT_IMAGE}")

    print(f"Processing multi_pe_rpe anatomical reference")
    print(f"Input: {INPUT_IMAGE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target resolution: {TARGET_VOXEL}mm isotropic")
    print()

    # Brain extraction and segmentation
    anat_outputs = brain_extract_and_segment(INPUT_IMAGE, OUTPUT_DIR)

    # Downsample MPRAGE, brain mask, and segmentation
    print("  Downsampling outputs...")
    for key, src_path in anat_outputs.items():
        stem = src_path.stem.replace('.nii', '')  # Handle .nii.gz
        suffix = f"_downsampled_{str(TARGET_VOXEL).replace('.', 'p')}"
        dst_name = f"{stem}{suffix}.nii.gz"
        dst_path = OUTPUT_DIR / dst_name

        # Use nearest-neighbor for masks and segmentations (label maps)
        interp = "nearest" if ("mask" in key or "segmentation" in key) else "sinc"
        print(f"    {src_path.name} ({interp}) -> {dst_name}")
        downsample_image(src_path, dst_path, TARGET_VOXEL, interp=interp)

    print(f"  Saved all outputs to: {OUTPUT_DIR}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
