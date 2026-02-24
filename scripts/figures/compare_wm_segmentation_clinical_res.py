#!/usr/bin/env python3
"""Compare white matter segmentation between anatomical reference and DWI-based segmentation.

Evaluates anatomical fidelity by computing two metrics:
1. Dice coefficient: Overlap between binary masks (threshold = 0.5)
2. Overlap score: 1 - mean(abs(reference - test)) for voxels where reference > 0.5
   (metric from sia_overlap.m, measures probability map agreement)

Compares:
- Reference: MTsat/MPRAGE-based WM probability map (SPM c2, ground truth)
- Test: DWI-based WM probability map (from multi-channel segmentation with FA + S0)

Across different resolutions and acquisition modes (PE vs RPE).
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
SINGLE_PE_RPE_BASE = ROOT / "data/single_pe_rpe"
MULTI_PE_RPE_BASE = ROOT / "data/multi_pe_rpe"

SINGLE_RESOLUTIONS = ["native_res", "downsampled_2p5mm"]
SINGLE_RESOLUTION_LABELS = {"native_res": "Native (1.6mm)", "downsampled_2p5mm": "Downsampled (2.5mm)"}

MULTI_RESOLUTIONS = ["native_res", "downsampled_2p5mm"]
MULTI_RESOLUTION_LABELS = {"native_res": "Native (1.7mm)", "downsampled_2p5mm": "Downsampled (2.5mm)"}

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


def compute_overlap(prob_ref: np.ndarray, prob_test: np.ndarray, threshold: float = 0.5) -> float:
    """Compute overlap metric between two probability maps.

    This implements the metric from sia_overlap.m:
    overlap = 1 - mean(abs(reference - test)) for voxels where reference > threshold

    Parameters
    ----------
    prob_ref : np.ndarray
        Reference probability map
    prob_test : np.ndarray
        Test probability map
    threshold : float
        Threshold for selecting voxels (default: 0.5)

    Returns
    -------
    float
        Overlap score (0-1, higher is better)
    """
    # Select voxels where reference probability > threshold
    mask = prob_ref > threshold

    if np.sum(mask) == 0:
        return 1.0  # No reference voxels

    # Compute mean absolute difference within masked region
    mean_abs_diff = np.mean(np.abs(prob_ref[mask] - prob_test[mask]))

    # Convert to overlap score (1 = perfect agreement, 0 = maximum disagreement)
    overlap = 1.0 - mean_abs_diff

    return overlap


def load_prob_map(prob_map_path: Path) -> np.ndarray:
    """Load probability map without thresholding.

    Parameters
    ----------
    prob_map_path : Path
        Path to probability map NIfTI file

    Returns
    -------
    np.ndarray
        Probability map (float array, range 0-1)
    """
    img = nib.load(str(prob_map_path))
    prob_data = img.get_fdata()
    return prob_data


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


def find_dwi_wm_maps(subject_dir: Path, resolution: str) -> dict[str, Path]:
    """Find DWI-based WM probability maps for a subject at a given resolution.

    Parameters
    ----------
    subject_dir : Path
        Subject directory name (e.g., "sub-V06460")
    resolution : str
        Resolution identifier (e.g., "native_res", "downsampled_2p5mm")

    Returns
    -------
    dict[str, Path]
        Dictionary with keys "PE" and "RPE" mapping to file paths (if they exist)
    """
    dwi_dir = SINGLE_PE_RPE_BASE / resolution / "processed" / subject_dir.name / "dwi"

    if not dwi_dir.exists():
        return {}

    subject_id = subject_dir.name
    results = {}

    # PE (single phase encoding): dir-AP
    pe_pattern = f"{subject_id}_*dir-AP*_preprocessed_wm_in_anat.nii.gz"
    pe_files = list(dwi_dir.glob(pe_pattern))
    if pe_files:
        results["PE"] = pe_files[0]

    # RPE (reverse phase encoding): merged
    rpe_pattern = f"{subject_id}_*merged*_preprocessed_wm_in_anat.nii.gz"
    rpe_files = list(dwi_dir.glob(rpe_pattern))
    if rpe_files:
        results["RPE"] = rpe_files[0]

    return results


def collect_dice_scores_single_pe_rpe() -> pd.DataFrame:
    """Collect Dice and Overlap scores for single_pe_rpe dataset (multi-subject).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Subject, Resolution, Mode, Dice, Overlap
    """
    # Find all subjects in native_res
    native_processed = SINGLE_PE_RPE_BASE / "native_res" / "processed"
    subjects = sorted([d for d in native_processed.iterdir()
                      if d.is_dir() and d.name.startswith("sub-")])

    results = []

    for subject_dir in subjects:
        subject_id = subject_dir.name
        print(f"Processing {subject_id}...")

        # Reference: MTsat-based WM segmentation (SPM c2 = WM)
        ref_wm_path = subject_dir / "mpm" / f"c2{subject_id}_MTsat.nii"

        if not ref_wm_path.exists():
            print(f"  Warning: Reference WM map not found: {ref_wm_path}")
            continue

        # Load reference probability map and binary mask
        try:
            ref_prob = load_prob_map(ref_wm_path)
            ref_mask = ref_prob >= WM_PROB_THRESHOLD
        except Exception as e:
            print(f"  Error loading reference: {e}")
            continue

        # Process each resolution
        for resolution in SINGLE_RESOLUTIONS:
            dwi_wm_maps = find_dwi_wm_maps(subject_dir, resolution)

            if not dwi_wm_maps:
                print(f"  No DWI WM maps found for {resolution}")
                continue

            # Process each mode (PE, RPE)
            for mode, dwi_wm_path in dwi_wm_maps.items():
                try:
                    # Load DWI-based probability map and binary mask
                    dwi_prob = load_prob_map(dwi_wm_path)
                    dwi_mask = dwi_prob >= WM_PROB_THRESHOLD

                    # Compute both metrics
                    dice = compute_dice(ref_mask, dwi_mask)
                    overlap = compute_overlap(ref_prob, dwi_prob, WM_PROB_THRESHOLD)

                    results.append({
                        "Subject": subject_id,
                        "Resolution": SINGLE_RESOLUTION_LABELS[resolution],
                        "Mode": mode,
                        "Dice": dice,
                        "Overlap": overlap
                    })

                    print(f"  {resolution} - {mode}: Dice = {dice:.4f}, Overlap = {overlap:.4f}")

                except Exception as e:
                    print(f"  Error processing {resolution} - {mode}: {e}")

        print()

    df = pd.DataFrame(results)
    return df


def collect_dice_scores_multi_pe_rpe() -> pd.DataFrame:
    """Collect Dice and Overlap scores for multi_pe_rpe dataset (single-subject).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Resolution, Mode, Dice, Overlap
    """
    print("Processing multi_pe_rpe dataset...")

    # Reference: MPRAGE-based WM segmentation (SPM c2 = WM)
    ref_wm_path = MULTI_PE_RPE_BASE / "native_res" / "processed" / "anat" / "c2mprage.nii"

    if not ref_wm_path.exists():
        print(f"  Warning: Reference WM map not found: {ref_wm_path}")
        return pd.DataFrame()

    # Load reference probability map and binary mask
    try:
        ref_prob = load_prob_map(ref_wm_path)
        ref_mask = ref_prob >= WM_PROB_THRESHOLD
    except Exception as e:
        print(f"  Error loading reference: {e}")
        return pd.DataFrame()

    results = []

    # Process each resolution
    for resolution in MULTI_RESOLUTIONS:
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
                # Load DWI-based probability map and binary mask
                dwi_prob = load_prob_map(dwi_wm_path)
                dwi_mask = dwi_prob >= WM_PROB_THRESHOLD

                # Compute both metrics
                dice = compute_dice(ref_mask, dwi_mask)
                overlap = compute_overlap(ref_prob, dwi_prob, WM_PROB_THRESHOLD)

                results.append({
                    "Resolution": MULTI_RESOLUTION_LABELS[resolution],
                    "Mode": mode,
                    "Dice": dice,
                    "Overlap": overlap
                })

                print(f"  {resolution} - {mode}: Dice = {dice:.4f}, Overlap = {overlap:.4f}")

            except Exception as e:
                print(f"  Error processing {resolution} - {mode}: {e}")

    print()

    df = pd.DataFrame(results)
    return df


def plot_dice_scores(df_single: pd.DataFrame, df_multi: pd.DataFrame, output_path: Path):
    """Create 2x2 grid comparing Dice and Overlap scores across both datasets.

    Parameters
    ----------
    df_single : pd.DataFrame
        Single_pe_rpe DataFrame with columns: Subject, Resolution, Mode, Dice, Overlap
    df_multi : pd.DataFrame
        Multi_pe_rpe DataFrame with columns: Resolution, Mode, Dice, Overlap
    output_path : Path
        Output path for the figure
    """
    # Set up the plot style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    hue_order = ["PE", "RPE"]
    palette = ["#E69F00", "#56B4E9"]  # Colorblind-friendly palette

    # Row 1: Dice coefficients
    # Subplot (0,0): single_pe_rpe Dice
    if not df_single.empty:
        x_order_single = list(SINGLE_RESOLUTION_LABELS.values())
        sns.barplot(
            data=df_single,
            x="Resolution",
            y="Dice",
            hue="Mode",
            order=x_order_single,
            hue_order=hue_order,
            palette=palette,
            errorbar="sd",  # Show standard deviation as error bars
            ax=axes[0, 0]
        )
        axes[0, 0].set_xlabel("Resolution", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Dice Coefficient", fontsize=12, fontweight="bold")
        axes[0, 0].set_title("Single PE/RPE Dataset - Dice\n(MTsat reference)",
                            fontsize=14, fontweight="bold", pad=20)
        axes[0, 0].set_ylim(0, 1.0)
        axes[0, 0].legend(title="Acquisition", title_fontsize=11, fontsize=10, loc="lower right")
        axes[0, 0].grid(axis="y", alpha=0.3)
        axes[0, 0].axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Subplot (0,1): multi_pe_rpe Dice
    if not df_multi.empty:
        x_order_multi = list(MULTI_RESOLUTION_LABELS.values())
        sns.barplot(
            data=df_multi,
            x="Resolution",
            y="Dice",
            hue="Mode",
            order=x_order_multi,
            hue_order=hue_order,
            palette=palette,
            errorbar=None,  # No error bars for single subject
            ax=axes[0, 1]
        )
        axes[0, 1].set_xlabel("Resolution", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Dice Coefficient", fontsize=12, fontweight="bold")
        axes[0, 1].set_title("Multi PE/RPE Dataset - Dice\n(MPRAGE reference)",
                            fontsize=14, fontweight="bold", pad=20)
        axes[0, 1].set_ylim(0, 1.0)
        axes[0, 1].legend(title="Acquisition", title_fontsize=11, fontsize=10, loc="lower right")
        axes[0, 1].grid(axis="y", alpha=0.3)
        axes[0, 1].axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Row 2: Overlap scores
    # Subplot (1,0): single_pe_rpe Overlap
    if not df_single.empty:
        x_order_single = list(SINGLE_RESOLUTION_LABELS.values())
        sns.barplot(
            data=df_single,
            x="Resolution",
            y="Overlap",
            hue="Mode",
            order=x_order_single,
            hue_order=hue_order,
            palette=palette,
            errorbar="sd",
            ax=axes[1, 0]
        )
        axes[1, 0].set_xlabel("Resolution", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Overlap Score", fontsize=12, fontweight="bold")
        axes[1, 0].set_title("Single PE/RPE Dataset - Overlap\n(MTsat reference)",
                            fontsize=14, fontweight="bold", pad=20)
        axes[1, 0].set_ylim(0, 1.0)
        axes[1, 0].legend(title="Acquisition", title_fontsize=11, fontsize=10, loc="lower right")
        axes[1, 0].grid(axis="y", alpha=0.3)
        axes[1, 0].axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Subplot (1,1): multi_pe_rpe Overlap
    if not df_multi.empty:
        x_order_multi = list(MULTI_RESOLUTION_LABELS.values())
        sns.barplot(
            data=df_multi,
            x="Resolution",
            y="Overlap",
            hue="Mode",
            order=x_order_multi,
            hue_order=hue_order,
            palette=palette,
            errorbar=None,
            ax=axes[1, 1]
        )
        axes[1, 1].set_xlabel("Resolution", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Overlap Score", fontsize=12, fontweight="bold")
        axes[1, 1].set_title("Multi PE/RPE Dataset - Overlap\n(MPRAGE reference)",
                            fontsize=14, fontweight="bold", pad=20)
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].legend(title="Acquisition", title_fontsize=11, fontsize=10, loc="lower right")
        axes[1, 1].grid(axis="y", alpha=0.3)
        axes[1, 1].axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot: {output_path}")


def print_summary_statistics(df_single: pd.DataFrame, df_multi: pd.DataFrame):
    """Print summary statistics of Dice and Overlap scores for both datasets.

    Parameters
    ----------
    df_single : pd.DataFrame
        Single_pe_rpe DataFrame with columns: Subject, Resolution, Mode, Dice, Overlap
    df_multi : pd.DataFrame
        Multi_pe_rpe DataFrame with columns: Resolution, Mode, Dice, Overlap
    """
    from scipy.stats import ttest_rel

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Single PE/RPE dataset
    if not df_single.empty:
        print("\n--- Single PE/RPE Dataset (Multi-Subject) ---")

        # Dice statistics
        print("\n** DICE COEFFICIENT **")
        grouped = df_single.groupby(["Resolution", "Mode"])["Dice"]
        summary = grouped.agg(["count", "mean", "std", "min", "max"])
        summary.columns = ["N", "Mean", "Std", "Min", "Max"]
        print(summary.to_string())

        # Overlap statistics
        print("\n** OVERLAP SCORE **")
        grouped = df_single.groupby(["Resolution", "Mode"])["Overlap"]
        summary = grouped.agg(["count", "mean", "std", "min", "max"])
        summary.columns = ["N", "Mean", "Std", "Min", "Max"]
        print(summary.to_string())

        # Statistical comparison between PE and RPE
        print("\n" + "-" * 70)
        print("PE vs RPE Comparison (paired t-test)")
        print("-" * 70)

        for metric in ["Dice", "Overlap"]:
            print(f"\n** {metric.upper()} **")
            for resolution in df_single["Resolution"].unique():
                res_df = df_single[df_single["Resolution"] == resolution]

                # Get paired data (same subjects)
                pe_data = res_df[res_df["Mode"] == "PE"].set_index("Subject")[metric]
                rpe_data = res_df[res_df["Mode"] == "RPE"].set_index("Subject")[metric]

                # Find common subjects
                common_subjects = pe_data.index.intersection(rpe_data.index)

                if len(common_subjects) > 1:
                    pe_paired = pe_data.loc[common_subjects]
                    rpe_paired = rpe_data.loc[common_subjects]

                    t_stat, p_value = ttest_rel(pe_paired, rpe_paired)

                    print(f"\n{resolution}:")
                    print(f"  N pairs: {len(common_subjects)}")
                    print(f"  PE mean:  {pe_paired.mean():.4f} ± {pe_paired.std():.4f}")
                    print(f"  RPE mean: {rpe_paired.mean():.4f} ± {rpe_paired.std():.4f}")
                    print(f"  Difference: {rpe_paired.mean() - pe_paired.mean():.4f}")
                    print(f"  t-statistic: {t_stat:.4f}")
                    print(f"  p-value: {p_value:.4f}")

                    if p_value < 0.001:
                        sig_str = "***"
                    elif p_value < 0.01:
                        sig_str = "**"
                    elif p_value < 0.05:
                        sig_str = "*"
                    else:
                        sig_str = "n.s."
                    print(f"  Significance: {sig_str}")

    # Multi PE/RPE dataset
    if not df_multi.empty:
        print("\n\n--- Multi PE/RPE Dataset (Single-Subject) ---")

        # Dice statistics
        print("\n** DICE COEFFICIENT **")
        grouped = df_multi.groupby(["Resolution", "Mode"])["Dice"]
        summary = grouped.agg(["count", "mean"])
        summary.columns = ["N", "Dice"]
        print(summary.to_string())

        # Overlap statistics
        print("\n** OVERLAP SCORE **")
        grouped = df_multi.groupby(["Resolution", "Mode"])["Overlap"]
        summary = grouped.agg(["count", "mean"])
        summary.columns = ["N", "Overlap"]
        print(summary.to_string())

        # Simple difference (no t-test for single subject)
        print("\n" + "-" * 70)
        print("PE vs RPE Comparison")
        print("-" * 70)

        for metric in ["Dice", "Overlap"]:
            print(f"\n** {metric.upper()} **")
            for resolution in df_multi["Resolution"].unique():
                res_df = df_multi[df_multi["Resolution"] == resolution]
                pe_value = res_df[res_df["Mode"] == "PE"][metric].values
                rpe_value = res_df[res_df["Mode"] == "RPE"][metric].values

                if len(pe_value) > 0 and len(rpe_value) > 0:
                    print(f"\n{resolution}:")
                    print(f"  PE:  {pe_value[0]:.4f}")
                    print(f"  RPE: {rpe_value[0]:.4f}")
                    print(f"  Difference: {rpe_value[0] - pe_value[0]:.4f}")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main function to compare WM segmentation and generate plots."""
    print("Comparing white matter segmentation: DWI vs Anatomical Reference")
    print(f"Probability threshold: {WM_PROB_THRESHOLD}")
    print("=" * 60)
    print()

    # Collect Dice scores from both datasets
    print("=== Single PE/RPE Dataset ===")
    df_single = collect_dice_scores_single_pe_rpe()
    print()

    print("=== Multi PE/RPE Dataset ===")
    df_multi = collect_dice_scores_multi_pe_rpe()
    print()

    if df_single.empty and df_multi.empty:
        print("No data collected from either dataset. Exiting.")
        return

    # Create output directory
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary statistics
    print_summary_statistics(df_single, df_multi)

    # Create plot
    plot_path = output_dir / "wm_segmentation_comparison_clinical_res.png"
    plot_dice_scores(df_single, df_multi, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
