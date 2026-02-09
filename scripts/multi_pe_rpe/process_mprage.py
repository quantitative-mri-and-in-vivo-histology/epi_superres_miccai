"""Process MPRAGE (T1w) image: brain extraction via FSL BET.

Uses ANTsPy for denoising and N4 bias field correction, and FSL BET
for brain extraction.
"""

import subprocess
from pathlib import Path

import ants

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_IMAGE = ROOT / "data/multi_pe_rpe/native_res/raw/anat/mprage.nii.gz"
OUTPUT_DIR = ROOT / "data/multi_pe_rpe/native_res/processed/anat"
TARGET_VOXEL = 1.7


def process_mprage(
    input_image: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Process MPRAGE: denoise, N4, brain extract (BET), segment (FAST).

    Pipeline:
    1. Denoise (non-local means)
    2. N4 bias field correction
    3. Brain extraction (FSL BET)
    4. Tissue segmentation (FSL FAST): 1=CSF, 2=GM, 3=WM
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Loading: {input_image}")
    t1 = ants.image_read(str(input_image))

    # Denoise
    print("Denoising...")
    t1_denoised = ants.denoise_image(t1)

    # N4 bias field correction
    print("N4 bias field correction...")
    t1_n4 = ants.n4_bias_field_correction(t1_denoised)

    mprage_path = output_dir / "mprage.nii.gz"
    ants.image_write(t1_n4, str(mprage_path))
    print(f"Saved corrected MPRAGE: {mprage_path}")

    # Brain extraction with FSL BET
    print("Brain extraction (FSL BET)...")
    brain_path = output_dir / "mprage_brain.nii.gz"
    brain_mask_path = output_dir / "brain_mask.nii.gz"
    subprocess.run(
        ["bet", str(mprage_path), str(brain_path), "-m", "-R"],
        check=True,
    )
    # BET writes mask as <output>_mask.nii.gz
    bet_mask_path = output_dir / "mprage_brain_mask.nii.gz"
    bet_mask_path.rename(brain_mask_path)
    print(f"Saved brain mask: {brain_mask_path}")
    print(f"Saved brain-extracted MPRAGE: {brain_path}")

    # Tissue segmentation with FSL FAST (1=CSF, 2=GM, 3=WM)
    print("Tissue segmentation (FSL FAST)...")
    fast_prefix = str(output_dir / "fast")
    subprocess.run(
        ["fast", "-o", fast_prefix, str(brain_path)],
        check=True,
    )
    # FAST outputs: fast_seg.nii.gz, fast_pve_0/1/2.nii.gz, etc.
    seg_path = output_dir / "segmentation.nii.gz"
    Path(f"{fast_prefix}_seg.nii.gz").rename(seg_path)
    print(f"Saved segmentation: {seg_path}")

    return {
        "mprage": mprage_path,
        "brain_mask": brain_mask_path,
        "brain": brain_path,
        "segmentation": seg_path,
    }


def downsample(native_outputs: dict[str, Path], output_dir: Path, voxel: float):
    """Downsample native outputs to target isotropic resolution.

    Uses sinc interpolation for images, nearest-neighbour for masks.
    Adds _downsampled_1p7 suffix to filenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, src in native_outputs.items():
        # Add suffix before extension
        stem = src.stem.replace('.nii', '')  # Handle .nii.gz
        suffix = f"_downsampled_{str(voxel).replace('.', 'p')}"
        dst_name = f"{stem}{suffix}.nii.gz"
        dst = output_dir / dst_name

        interp = "nearest" if "mask" in key else "sinc"
        print(f"Downsampling {src.name} ({interp}) -> {dst_name}")
        subprocess.run(
            ["mrgrid", str(src), "regrid", str(dst),
            "-voxel", str(voxel), "-interp", interp, "-force"],
            check=True,
        )

    # Re-threshold mask to binary after regrid (safety)
    mask_dst = output_dir / f"brain_mask_downsampled_{str(voxel).replace('.', 'p')}.nii.gz"
    if mask_dst.exists():
        mask = ants.image_read(str(mask_dst))
        mask = ants.threshold_image(mask, low_thresh=0.5)
        ants.image_write(mask, str(mask_dst))


if __name__ == "__main__":
    print("Processing MPRAGE")
    print(f"Input:  {INPUT_IMAGE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    outputs = process_mprage(INPUT_IMAGE, OUTPUT_DIR)

    print()
    print(f"Downsampling to {TARGET_VOXEL}mm isotropic...")
    downsample(outputs, OUTPUT_DIR, TARGET_VOXEL)

    print()
    print("Done")
