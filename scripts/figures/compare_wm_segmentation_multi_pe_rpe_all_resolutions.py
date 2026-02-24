#!/usr/bin/env python3
"""Compare white matter segmentation across all resolutions for multi_pe_rpe dataset.

Evaluates anatomical fidelity by computing Dice coefficient between:
- Reference: MPRAGE-based WM probability map (ground truth)
- Test: DWI-based WM probability map (from multi-channel Atropos with FA + S0)

Compares PE vs RPE modes across all available resolutions.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

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

# Probability threshold for creating binary masks
WM_PROB_THRESHOLD = 0.5


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask
    mask2 : np.ndarray
        Second binary mask

    Returns
    -------
    float
        Dice coefficient (0-1)
    """
    intersection = np.sum(mask1 * mask2)
    sum_masks = np.sum(mask1) + np.sum(mask2)

    if sum_masks == 0:
        return 1.0  # Both masks empty

    dice = 2.0 * intersection / sum_masks
    return dice


def load_and_threshold_prob_map(prob_map_path: Path, threshold: float = 0.5) -> np.ndarray:
    """Load probability map and threshold to binary mask.

    Parameters
    ----------
    prob_map_path : Path
        Path to probability map NIfTI file
    threshold : float
        Threshold for binarization (default: 0.5)

    Returns
    -------
    np.ndarray
        Binary mask (boolean array)
    """
    img = nib.load(str(prob_map_path))
    prob_data = img.get_fdata()
    binary_mask = prob_data >= threshold
    return binary_mask


def collect_dice_scores() -> pd.DataFrame:
    """Collect Dice scores for multi_pe_rpe dataset across all resolutions.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Resolution, Mode, Dice
    """
    print("Processing multi_pe_rpe dataset across all resolutions...")

    # Reference: MPRAGE-based WM segmentation (SPM c2)
    ref_wm_path = MULTI_PE_RPE_BASE / "native_res" / "processed" / "anat" / "c2mprage.nii"

    if not ref_wm_path.exists():
        print(f"  Warning: Reference WM map not found: {ref_wm_path}")
        return pd.DataFrame()

    # Load reference mask
    try:
        ref_mask = load_and_threshold_prob_map(ref_wm_path, WM_PROB_THRESHOLD)
    except Exception as e:
        print(f"  Error loading reference: {e}")
        return pd.DataFrame()

    results = []

    # Process each resolution
    for resolution in RESOLUTIONS:
        dwi_dir = MULTI_PE_RPE_BASE / resolution / "processed" / "dwi"

        if not dwi_dir.exists():
            print(f"  No DWI directory found for {resolution}")
            continue

        # PE mode: dwi_lr
        pe_wm_path = dwi_dir / "dwi_lr_preprocessed_wm_in_anat.nii.gz"
        # RPE mode: dwi_merged
        rpe_wm_path = dwi_dir / "dwi_merged_preprocessed_wm_in_anat.nii.gz"

        for mode, dwi_wm_path in [("PE", pe_wm_path), ("RPE", rpe_wm_path)]:
            if not dwi_wm_path.exists():
                print(f"  No DWI WM map found for {resolution} - {mode}")
                continue

            try:
                # Load DWI-based mask
                dwi_mask = load_and_threshold_prob_map(dwi_wm_path, WM_PROB_THRESHOLD)

                # Compute Dice coefficient
                dice = compute_dice(ref_mask, dwi_mask)

                results.append({
                    "Resolution": RESOLUTION_LABELS[resolution],
                    "Mode": mode,
                    "Dice": dice
                })

                print(f"  {resolution} - {mode}: Dice = {dice:.4f}")

            except Exception as e:
                print(f"  Error processing {resolution} - {mode}: {e}")

    print()

    df = pd.DataFrame(results)
    return df


def plot_dice_scores(df: pd.DataFrame, output_path: Path):
    """Create bar plot comparing Dice scores across all resolutions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Resolution, Mode, Dice
    output_path : Path
        Output path for the figure
    """
    # Set up the plot style
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    hue_order = ["PE", "RPE"]
    palette = ["#E69F00", "#56B4E9"]  # Colorblind-friendly palette

    x_order = list(RESOLUTION_LABELS.values())

    # Create bar plot
    sns.barplot(
        data=df,
        x="Resolution",
        y="Dice",
        hue="Mode",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        errorbar=None,  # No error bars for single subject
        ax=ax
    )

    ax.set_xlabel("Resolution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dice Coefficient", fontsize=12, fontweight="bold")
    ax.set_title("White Matter Segmentation: Multi PE/RPE Dataset\n(MPRAGE reference)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0.5, 1.0)
    ax.legend(title="Acquisition", title_fontsize=11, fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot: {output_path}")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of Dice scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Resolution, Mode, Dice
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if df.empty:
        print("No data available.")
        return

    print("\n--- Multi PE/RPE Dataset (Single-Subject) ---")
    grouped = df.groupby(["Resolution", "Mode"])["Dice"]
    summary = grouped.agg(["count", "mean"])
    summary.columns = ["N", "Dice"]
    print("\n", summary.to_string())

    # Simple difference (no t-test for single subject)
    print("\n" + "-" * 60)
    print("PE vs RPE Comparison")
    print("-" * 60)

    for resolution in df["Resolution"].unique():
        res_df = df[df["Resolution"] == resolution]
        pe_dice = res_df[res_df["Mode"] == "PE"]["Dice"].values
        rpe_dice = res_df[res_df["Mode"] == "RPE"]["Dice"].values

        if len(pe_dice) > 0 and len(rpe_dice) > 0:
            print(f"\n{resolution}:")
            print(f"  PE:  {pe_dice[0]:.4f}")
            print(f"  RPE: {rpe_dice[0]:.4f}")
            print(f"  Difference (RPE - PE): {rpe_dice[0] - pe_dice[0]:.4f}")

    print("\n" + "=" * 60 + "\n")


def main():
    """Main function to compare WM segmentation and generate plots."""
    print("Comparing white matter segmentation: Multi PE/RPE Dataset (All Resolutions)")
    print(f"Probability threshold: {WM_PROB_THRESHOLD}")
    print("=" * 60)
    print()

    # Collect Dice scores
    df = collect_dice_scores()

    if df.empty:
        print("No data collected. Exiting.")
        return

    # Create output directory
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary statistics
    print_summary_statistics(df)

    # Create plot
    plot_path = output_dir / "wm_segmentation_multi_pe_rpe_all_resolutions.png"
    plot_dice_scores(df, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
