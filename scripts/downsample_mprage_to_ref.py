#!/usr/bin/env python3
"""Downsample T1w image to 1.7mm isotropic using MRtrix mrgrid with sinc interpolation."""

from pathlib import Path

from utils.cmd_utils import run_command


ROOT = Path(__file__).parent.parent
INPUT = ROOT / "data" / "raw" / "invivo_highres_whole_brain" / "mprage.nii.gz"
OUTPUT_DIR = ROOT / "data" / "processed" / "invivo_highres_whole_brain" / "mprage"
TARGET_VOXEL = 1.7


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"T1w image not found: {INPUT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "mprage_1p7_iso.nii.gz"

    print(f"Input:  {INPUT}")
    print(f"Output: {output}")
    print(f"Target: {TARGET_VOXEL}mm isotropic")

    cmd = (
        f"mrgrid {INPUT} regrid {output} -force "
        f"-voxel {TARGET_VOXEL} "
        f"-interp sinc"
    )
    run_command(cmd)

    print("Done.")


if __name__ == "__main__":
    main()
