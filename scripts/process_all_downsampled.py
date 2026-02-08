#!/usr/bin/env python3
"""Process all downsampled DWI datasets with FSL preprocessing pipeline.

Automatically finds all downsampled_* directories and processes them.
"""

import re
import subprocess
import sys
from pathlib import Path


# Common parameters
BASE_DIR = Path(__file__).parent.parent
SCRIPT = BASE_DIR / "scripts" / "process_pe_fsl.py"
DWI_BASE_DIR = BASE_DIR / "data/8_pe/processed/dwi"
T1W_REF = BASE_DIR / "data/8_pe/processed/anat/downsampled_1p7mm/t1w_brain.nii.gz"
NTHREADS = 50


def find_downsampled_dirs() -> list[tuple[str, float]]:
    """Find all downsampled_* directories in the DWI base directory.

    Returns
    -------
    list[tuple[str, float]]
        List of (directory_name, factor) tuples, sorted by factor.
        E.g., [("downsampled_0p5", 0.5), ("downsampled_0p6", 0.6), ...]
    """
    if not DWI_BASE_DIR.exists():
        return []

    dirs = []
    pattern = re.compile(r"downsampled_(\d+)p(\d+)")

    for path in DWI_BASE_DIR.iterdir():
        if path.is_dir() and path.name.startswith("downsampled_"):
            match = pattern.match(path.name)
            if match:
                # Convert "0p6" to 0.6
                integer_part = match.group(1)
                decimal_part = match.group(2)
                factor = float(f"{integer_part}.{decimal_part}")
                dirs.append((path.name, factor))

    # Sort by factor
    dirs.sort(key=lambda x: x[1])
    return dirs


def process_factor(dir_name: str, factor: float) -> int:
    """Process a single downsampled directory.

    Parameters
    ----------
    dir_name : str
        Directory name (e.g., "downsampled_0p6")
    factor : float
        Downsampling factor (e.g., 0.6)

    Returns
    -------
    int
        Return code from the process (0 = success)
    """
    # Set up paths
    dwi_dir = DWI_BASE_DIR / dir_name
    dwi_lr = dwi_dir / "raw/dwi_lr.nii.gz"
    dwi_rl = dwi_dir / "raw/dwi_rl.nii.gz"
    output_dir = dwi_dir

    # Check if input files exist
    if not dwi_lr.exists() or not dwi_rl.exists():
        print(f"⚠️  Skipping {dir_name}: input files not found")
        print(f"    Expected: {dwi_lr}")
        print(f"              {dwi_rl}")
        return 1

    print(f"\n{'='*80}")
    print(f"Processing {dir_name} (factor={factor:.1f})")
    print(f"{'='*80}")

    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        str(SCRIPT),
        "--dwi", str(dwi_lr),
        "--dwi-rpe", str(dwi_rl),
        "--output-dir", str(output_dir),
        "--t1w", str(T1W_REF),
        "--keep-tmp",
        "--nthreads", str(NTHREADS),
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Run the command
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"✓ Successfully processed {dir_name}")
    else:
        print(f"✗ Failed to process {dir_name} (exit code: {result.returncode})")

    return result.returncode


def main():
    """Process all downsampled directories."""
    print("Processing all downsampled DWI datasets")
    print(f"Searching for downsampled_* directories in: {DWI_BASE_DIR}")

    # Find all downsampled directories
    downsampled_dirs = find_downsampled_dirs()

    if not downsampled_dirs:
        print(f"\n⚠️  No downsampled_* directories found in {DWI_BASE_DIR}")
        sys.exit(1)

    print(f"Found {len(downsampled_dirs)} downsampled dataset(s):")
    for dir_name, factor in downsampled_dirs:
        print(f"  - {dir_name} (factor={factor:.1f})")

    print(f"\nT1w reference: {T1W_REF}")
    print(f"Threads: {NTHREADS}")

    # Check if T1w reference exists
    if not T1W_REF.exists():
        print(f"\n⚠️  WARNING: T1w reference not found: {T1W_REF}")
        print("Processing will continue without T1w registration.")

    failed = []

    # Process each directory
    for dir_name, factor in downsampled_dirs:
        exit_code = process_factor(dir_name, factor)
        if exit_code != 0:
            failed.append(dir_name)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(downsampled_dirs)} datasets")
    print(f"Success: {len(downsampled_dirs) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed datasets: {failed}")
        sys.exit(1)
    else:
        print("\n✓ All datasets processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
