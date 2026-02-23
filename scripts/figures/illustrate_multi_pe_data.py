#!/usr/bin/env python3
"""Illustrate multi-PE data by visualizing all 8 phase encoding directions.

Shows the first volume (b=0) from each of the 8 phase encoding directions,
arranged as 4 reverse phase encoding pairs in a 2x4 grid.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Dataset configuration
MULTI_PE_RPE_RAW = ROOT / "data/multi_pe_rpe/native_res/raw/dwi"

# Phase encoding pairs (each pair is opposite directions)
PE_PAIRS = [
    ("dwi_lr.nii.gz", "dwi_rl.nii.gz", "LR ↔ RL"),
    ("dwi_ppos90.nii.gz", "dwi_pneg90.nii.gz", "P90° ↔ P-90°"),
    ("dwi_ppos45.nii.gz", "dwi_pneg135.nii.gz", "P45° ↔ P-135°"),
    ("dwi_ppos135.nii.gz", "dwi_pneg45.nii.gz", "P135° ↔ P-45°")
]


def load_first_volume(dwi_path: Path) -> np.ndarray:
    """Load first volume (b=0) from 4D DWI image.

    Parameters
    ----------
    dwi_path : Path
        Path to 4D DWI NIfTI file

    Returns
    -------
    np.ndarray
        3D volume (first volume from 4D series)
    """
    img = nib.load(str(dwi_path))
    data = img.get_fdata()

    # Extract first volume
    if data.ndim == 4:
        return data[:, :, :, 0]
    else:
        return data


def extract_mid_axial_slice(volume: np.ndarray) -> np.ndarray:
    """Extract middle axial slice (along third/last axis).

    Parameters
    ----------
    volume : np.ndarray
        3D volume

    Returns
    -------
    np.ndarray
        2D slice
    """
    mid_idx = volume.shape[2] // 2
    return volume[:, :, mid_idx]


def main():
    """Generate multi-PE illustration figure."""
    print("Illustrating multi-PE data (8 phase encoding directions)")
    print("=" * 60)

    # Check if directory exists
    if not MULTI_PE_RPE_RAW.exists():
        print(f"Error: Raw DWI directory not found: {MULTI_PE_RPE_RAW}")
        return

    # Load all images and extract slices
    pair_slices = []

    for pe1_name, pe2_name, pair_label in PE_PAIRS:
        pe1_path = MULTI_PE_RPE_RAW / pe1_name
        pe2_path = MULTI_PE_RPE_RAW / pe2_name

        if not pe1_path.exists():
            print(f"Warning: {pe1_name} not found")
            continue
        if not pe2_path.exists():
            print(f"Warning: {pe2_name} not found")
            continue

        # Load first volumes
        pe1_vol = load_first_volume(pe1_path)
        pe2_vol = load_first_volume(pe2_path)

        # Extract middle axial slices
        pe1_slice = extract_mid_axial_slice(pe1_vol)
        pe2_slice = extract_mid_axial_slice(pe2_vol)

        pair_slices.append((pe1_slice, pe2_slice, pair_label))

        print(f"Loaded {pair_label}: {pe1_name} and {pe2_name}")
        print(f"  Volume shape: {pe1_vol.shape}, Slice shape: {pe1_slice.shape}")

    if not pair_slices:
        print("No data found. Exiting.")
        return

    # Determine consistent color range using all data
    all_slices = []
    for pe1_slice, pe2_slice, _ in pair_slices:
        all_slices.extend([pe1_slice, pe2_slice])

    all_data = np.concatenate([s.flatten() for s in all_slices])
    all_data = all_data[all_data > 0]  # Exclude background

    vmin = np.percentile(all_data, 2)
    vmax = np.percentile(all_data, 98)

    print(f"\nColor range: [{vmin:.0f}, {vmax:.0f}]")

    # Create figure with 2 rows x 4 columns
    n_pairs = len(pair_slices)

    # Calculate figure size to preserve aspect ratio
    # Each slice is square (120x120), so we want square subplots
    fig, axes = plt.subplots(2, n_pairs, figsize=(n_pairs * 3, 6))

    # Plot each pair
    for col, (pe1_slice, pe2_slice, pair_label) in enumerate(pair_slices):
        # Top row: first direction
        ax_top = axes[0, col]
        ax_top.imshow(pe1_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
        ax_top.axis('off')

        # Bottom row: opposite direction
        ax_bot = axes[1, col]
        ax_bot.imshow(pe2_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')
        ax_bot.axis('off')

    # Tight layout with minimal spacing
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0, right=1, top=1, bottom=0)

    # Save figure in both PNG and SVG formats
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path_png = output_dir / "multi_pe_illustration.png"
    output_path_svg = output_dir / "multi_pe_illustration.svg"

    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight')

    print(f"\nSaved figures:")
    print(f"  PNG: {output_path_png}")
    print(f"  SVG: {output_path_svg}")

    print("\nDone!")


if __name__ == "__main__":
    main()
