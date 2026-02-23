#!/usr/bin/env python3
"""Compare FA maps across resolutions using native resolution as reference.

Evaluates FA quality by computing voxel-wise metrics between:
- Reference: Native resolution FA map (1.7mm, ground truth)
- Test: Downsampled resolution FA maps (2.0, 2.5, 3.0, 3.4mm)

Compares PE vs RPE modes across all downsampled resolutions.
Brain masks are applied before computing metrics.
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

# Only downsampled resolutions (native is the reference)
RESOLUTIONS = ["downsampled_2p0mm", "downsampled_2p5mm", "downsampled_3p0mm", "downsampled_3p4mm"]
RESOLUTION_LABELS = {
    "downsampled_2p0mm": "2.0mm",
    "downsampled_2p5mm": "2.5mm",
    "downsampled_3p0mm": "3.0mm",
    "downsampled_3p4mm": "3.4mm"
}


def load_and_mask_fa(fa_path: Path, mask_path: Path) -> np.ndarray:
    """Load FA map and apply brain mask.

    Parameters
    ----------
    fa_path : Path
        Path to FA NIfTI file
    mask_path : Path
        Path to brain mask NIfTI file

    Returns
    -------
    np.ndarray
        Masked FA data (only brain voxels)
    """
    fa_img = nib.load(str(fa_path))
    fa_data = fa_img.get_fdata()

    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata()

    # Apply mask (only keep brain voxels)
    masked_fa = fa_data[mask_data > 0]

    return masked_fa


def compute_fa_metrics(ref_fa: np.ndarray, test_fa: np.ndarray) -> dict[str, float]:
    """Compute voxel-wise metrics between reference and test FA maps.

    Parameters
    ----------
    ref_fa : np.ndarray
        Reference FA values (native resolution)
    test_fa : np.ndarray
        Test FA values (downsampled resolution)

    Returns
    -------
    dict[str, float]
        Dictionary with MAE, RMSE, and correlation
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(ref_fa - test_fa))

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((ref_fa - test_fa) ** 2))

    # Pearson correlation
    correlation = np.corrcoef(ref_fa, test_fa)[0, 1]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Correlation": correlation
    }


def collect_fa_metrics() -> pd.DataFrame:
    """Collect FA metrics for multi_pe_rpe dataset across downsampled resolutions.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Resolution, Mode, MAE, RMSE, Correlation
    """
    print("Comparing FA maps across resolutions (multi_pe_rpe)...")
    print()

    results = []

    # Process each mode (PE and RPE)
    for mode, mode_name in [("lr", "PE"), ("merged", "RPE")]:
        print(f"=== Mode: {mode_name} ===")

        # Reference: Native resolution FA and mask
        native_dwi_dir = MULTI_PE_RPE_BASE / "native_res" / "processed" / "dwi"
        ref_fa_path = native_dwi_dir / f"dwi_{mode}_preprocessed_anat_dti_fa.nii.gz"
        ref_mask_path = native_dwi_dir / f"dwi_{mode}_preprocessed_brain_mask.nii.gz"

        if not ref_fa_path.exists():
            print(f"  Warning: Reference FA not found: {ref_fa_path}")
            continue

        if not ref_mask_path.exists():
            print(f"  Warning: Reference mask not found: {ref_mask_path}")
            continue

        # Load reference FA (masked)
        try:
            ref_fa = load_and_mask_fa(ref_fa_path, ref_mask_path)
            print(f"  Loaded reference FA: {ref_fa_path.name}")
            print(f"    Brain voxels: {len(ref_fa)}")
            print(f"    FA range: [{ref_fa.min():.3f}, {ref_fa.max():.3f}]")
            print()
        except Exception as e:
            print(f"  Error loading reference: {e}")
            continue

        # Process each downsampled resolution
        for resolution in RESOLUTIONS:
            dwi_dir = MULTI_PE_RPE_BASE / resolution / "processed" / "dwi"

            test_fa_path = dwi_dir / f"dwi_{mode}_preprocessed_anat_dti_fa.nii.gz"
            test_mask_path = dwi_dir / f"dwi_{mode}_preprocessed_brain_mask.nii.gz"

            if not test_fa_path.exists():
                print(f"  {resolution}: FA not found")
                continue

            if not test_mask_path.exists():
                print(f"  {resolution}: Mask not found")
                continue

            try:
                # Load test FA (masked)
                test_fa = load_and_mask_fa(test_fa_path, test_mask_path)

                # Check if sizes match (they should, since both are in anat space)
                if len(ref_fa) != len(test_fa):
                    print(f"  {resolution}: Size mismatch (ref={len(ref_fa)}, test={len(test_fa)})")
                    continue

                # Compute metrics
                metrics = compute_fa_metrics(ref_fa, test_fa)

                results.append({
                    "Resolution": RESOLUTION_LABELS[resolution],
                    "Mode": mode_name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "Correlation": metrics["Correlation"]
                })

                print(f"  {resolution}:")
                print(f"    MAE:  {metrics['MAE']:.4f}")
                print(f"    RMSE: {metrics['RMSE']:.4f}")
                print(f"    Corr: {metrics['Correlation']:.4f}")

            except Exception as e:
                print(f"  Error processing {resolution}: {e}")

        print()

    df = pd.DataFrame(results)
    return df


def plot_fa_metrics(df: pd.DataFrame, output_path: Path):
    """Create multi-panel plot comparing FA metrics across resolutions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Resolution, Mode, MAE, RMSE, Correlation
    output_path : Path
        Output path for the figure
    """
    if df.empty:
        print("No data to plot.")
        return

    # Set up the plot style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    hue_order = ["PE", "RPE"]
    palette = ["#E69F00", "#56B4E9"]  # Colorblind-friendly palette
    x_order = list(RESOLUTION_LABELS.values())

    # Panel 1: MAE
    sns.barplot(
        data=df,
        x="Resolution",
        y="MAE",
        hue="Mode",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        errorbar=None,
        ax=axes[0]
    )
    axes[0].set_xlabel("Resolution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Mean Absolute Error", fontsize=12, fontweight="bold")
    axes[0].set_title("FA Mean Absolute Error\n(lower is better)",
                      fontsize=13, fontweight="bold", pad=15)
    axes[0].legend(title="Acquisition", title_fontsize=10, fontsize=9, loc="upper left")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: RMSE
    sns.barplot(
        data=df,
        x="Resolution",
        y="RMSE",
        hue="Mode",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        errorbar=None,
        ax=axes[1]
    )
    axes[1].set_xlabel("Resolution", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Root Mean Square Error", fontsize=12, fontweight="bold")
    axes[1].set_title("FA Root Mean Square Error\n(lower is better)",
                      fontsize=13, fontweight="bold", pad=15)
    axes[1].legend(title="Acquisition", title_fontsize=10, fontsize=9, loc="upper left")
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: Correlation
    sns.barplot(
        data=df,
        x="Resolution",
        y="Correlation",
        hue="Mode",
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        errorbar=None,
        ax=axes[2]
    )
    axes[2].set_xlabel("Resolution", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Pearson Correlation", fontsize=12, fontweight="bold")
    axes[2].set_title("FA Correlation with Native\n(higher is better)",
                      fontsize=13, fontweight="bold", pad=15)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(title="Acquisition", title_fontsize=10, fontsize=9, loc="lower left")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axes[2].text(0.02, 0.91, "High correlation (0.9)", fontsize=8, color="gray",
                 transform=axes[2].get_yaxis_transform())

    # Overall title
    fig.suptitle('FA Map Comparison Across Resolutions\n(Reference: Native 1.7mm)',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot: {output_path}")


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of FA metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Resolution, Mode, MAE, RMSE, Correlation
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if df.empty:
        print("No data available.")
        return

    print("\n--- FA Metrics vs Native Resolution (1.7mm) ---")

    # Group by resolution and mode
    for metric in ["MAE", "RMSE", "Correlation"]:
        print(f"\n{metric}:")
        pivot = df.pivot(index="Resolution", columns="Mode", values=metric)
        print(pivot.to_string())

    # PE vs RPE comparison
    print("\n" + "-" * 60)
    print("PE vs RPE Comparison")
    print("-" * 60)

    for resolution in df["Resolution"].unique():
        res_df = df[df["Resolution"] == resolution]
        pe_data = res_df[res_df["Mode"] == "PE"]
        rpe_data = res_df[res_df["Mode"] == "RPE"]

        if len(pe_data) > 0 and len(rpe_data) > 0:
            print(f"\n{resolution}:")
            print(f"  MAE:  PE={pe_data['MAE'].values[0]:.4f}, RPE={rpe_data['MAE'].values[0]:.4f}, Diff={rpe_data['MAE'].values[0] - pe_data['MAE'].values[0]:+.4f}")
            print(f"  RMSE: PE={pe_data['RMSE'].values[0]:.4f}, RPE={rpe_data['RMSE'].values[0]:.4f}, Diff={rpe_data['RMSE'].values[0] - pe_data['RMSE'].values[0]:+.4f}")
            print(f"  Corr: PE={pe_data['Correlation'].values[0]:.4f}, RPE={rpe_data['Correlation'].values[0]:.4f}, Diff={rpe_data['Correlation'].values[0] - pe_data['Correlation'].values[0]:+.4f}")

    print("\n" + "=" * 60 + "\n")


def main():
    """Main function to compare FA maps and generate plots."""
    print("Comparing FA maps across resolutions: Multi PE/RPE Dataset")
    print("Reference: Native resolution (1.7mm)")
    print("=" * 60)
    print()

    # Collect FA metrics
    df = collect_fa_metrics()

    if df.empty:
        print("No data collected. Exiting.")
        return

    # Create output directory
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary statistics
    print_summary_statistics(df)

    # Create plot
    plot_path = output_dir / "fa_comparison_across_resolutions.png"
    plot_fa_metrics(df, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
