#!/usr/bin/env python3
"""Compare FA and S0 maps across resolutions for multi_pe_rpe dataset.

Visualizes the effect of downsampling on diffusion tensor metrics by displaying
S0 (baseline signal) and FA (fractional anisotropy) maps across different resolutions
in a grid layout.
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
MULTI_PE_RPE_BASE = ROOT / "data/multi_pe_rpe"
RESOLUTIONS = ["native_res", "downsampled_2p0mm", "downsampled_2p5mm", "downsampled_3p0mm", "downsampled_3p4mm"]
RESOLUTION_LABELS = {
    "native_res": "Native (1.7mm)",
    "downsampled_2p0mm": "2.0mm",
    "downsampled_2p5mm": "2.5mm",
    "downsampled_3p0mm": "3.0mm",
    "downsampled_3p4mm": "3.4mm"
}


def load_image(file_path: Path) -> np.ndarray:
    """Load NIfTI image and return data array.

    Parameters
    ----------
    file_path : Path
        Path to NIfTI file

    Returns
    -------
    np.ndarray
        Image data
    """
    img = nib.load(str(file_path))
    return img.get_fdata()


def extract_middle_axial_slice(data: np.ndarray) -> np.ndarray:
    """Extract axial slice at 2/3 position from 3D volume.

    Parameters
    ----------
    data : np.ndarray
        3D image volume

    Returns
    -------
    np.ndarray
        2D axial slice
    """
    slice_idx = int(data.shape[2] * 2 / 3)
    return data[:, :, slice_idx]


def main():
    """Generate comparison figure of FA and S0 across resolutions."""
    print("Comparing FA and S0 across resolutions (multi_pe_rpe)")
    print("=" * 60)

    # Load all S0 and FA images
    s0_slices = []
    fa_slices = []
    valid_resolutions = []

    for resolution in RESOLUTIONS:
        dwi_dir = MULTI_PE_RPE_BASE / resolution / "processed" / "dwi"
        s0_path = dwi_dir / "dwi_merged_preprocessed_dti_s0.nii.gz"
        fa_path = dwi_dir / "dwi_merged_preprocessed_dti_fa.nii.gz"

        if not s0_path.exists() or not fa_path.exists():
            print(f"Warning: Missing files for {resolution}")
            print(f"  S0 exists: {s0_path.exists()}")
            print(f"  FA exists: {fa_path.exists()}")
            continue

        # Load images
        s0_data = load_image(s0_path)
        fa_data = load_image(fa_path)

        # Extract middle axial slice
        s0_slice = extract_middle_axial_slice(s0_data)
        fa_slice = extract_middle_axial_slice(fa_data)

        s0_slices.append(s0_slice)
        fa_slices.append(fa_slice)
        valid_resolutions.append(resolution)

        print(f"Loaded {resolution}: S0 shape={s0_slice.shape}, FA shape={fa_slice.shape}")

    if not s0_slices:
        print("No data found. Exiting.")
        return

    n_res = len(valid_resolutions)

    # Determine consistent color ranges using robust statistics
    # Combine all S0 data to compute global percentiles (more robust than per-image)
    all_s0_data = np.concatenate([s.flatten() for s in s0_slices])
    all_s0_data = all_s0_data[all_s0_data > 0]  # Exclude background (zero values)

    # Use 5th and 95th percentiles for more robust scaling
    s0_min = np.percentile(all_s0_data, 5)
    s0_max = np.percentile(all_s0_data, 95)

    fa_min = 0.0
    fa_max = 1.0

    print(f"\nColor ranges:")
    print(f"  S0: [{s0_min:.0f}, {s0_max:.0f}]")
    print(f"  FA: [{fa_min:.2f}, {fa_max:.2f}]")

    # Create figure with subplots (extra height for colorbars)
    fig, axes = plt.subplots(2, n_res, figsize=(4 * n_res, 10))

    # Ensure axes is 2D even with single resolution
    if n_res == 1:
        axes = axes.reshape(2, 1)

    # Plot S0 maps (top row)
    for i, (s0_slice, resolution) in enumerate(zip(s0_slices, valid_resolutions)):
        ax = axes[0, i]
        im_s0 = ax.imshow(s0_slice.T, cmap='gray', origin='lower',
                          vmin=s0_min, vmax=s0_max)
        ax.set_title(f"S0: {RESOLUTION_LABELS[resolution]}", fontsize=12, fontweight='bold')
        ax.axis('off')

    # Plot FA maps (bottom row)
    for i, (fa_slice, resolution) in enumerate(zip(fa_slices, valid_resolutions)):
        ax = axes[1, i]
        im_fa = ax.imshow(fa_slice.T, cmap='hot', origin='lower',
                          vmin=fa_min, vmax=fa_max)
        ax.set_title(f"FA: {RESOLUTION_LABELS[resolution]}", fontsize=12, fontweight='bold')
        ax.axis('off')

    # Add colorbars below each row with extra spacing
    # S0 colorbar (below top row)
    cbar_s0 = fig.colorbar(im_s0, ax=axes[0, :].tolist(), orientation='horizontal',
                           fraction=0.03, pad=0.25, aspect=40)
    cbar_s0.set_label('S0 (baseline signal intensity)', fontsize=11, fontweight='bold')

    # FA colorbar (below bottom row)
    cbar_fa = fig.colorbar(im_fa, ax=axes[1, :].tolist(), orientation='horizontal',
                           fraction=0.03, pad=0.25, aspect=40)
    cbar_fa.set_label('FA (fractional anisotropy)', fontsize=11, fontweight='bold')

    # Overall title
    fig.suptitle('DTI Metrics Across Resolutions\n(Multi PE/RPE Dataset)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=4.0)

    # Save figure
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fa_s0_resolution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
