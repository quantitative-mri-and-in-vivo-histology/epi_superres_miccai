#!/usr/bin/env python3
"""Downsample denoised DWI data using MRtrix mrgrid with sinc interpolation.

Creates synthetic low-resolution datasets at multiple downsampling factors
for super-resolution protocol optimization experiments.
"""

import argparse
from pathlib import Path
import shutil

from utils.cmd_utils import run_command
from utils.nifti_utils import get_bval_path, get_bvec_path, get_json_path


DOWNSAMPLE_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9]


def downsample_dwi(dwi: Path, output: Path, factor: float,
                    nthreads: int = None) -> Path:
    """Downsample a DWI volume using MRtrix mrgrid with sinc interpolation.

    Parameters
    ----------
    dwi : Path
        Input NIfTI file (.nii or .nii.gz).
    output : Path
        Output NIfTI file path.
    factor : float
        Resolution factor (0-1). E.g., 0.5 means half the resolution
        (voxel size doubled).
    nthreads : int, optional
        Number of threads for MRtrix.

    Returns
    -------
    Path
        Path to the downsampled output file.
    """
    cmd = (
        f"mrgrid {dwi} regrid {output} -force "
        f"-scale {factor},1,{factor} "
        f"-interp sinc"
    )
    if nthreads is not None:
        cmd += f" -nthreads {nthreads}"

    run_command(cmd)
    return output


def copy_associated_files(src: Path, dst: Path):
    """Copy bval, bvec, and JSON sidecar files alongside the output."""
    for get_path in [get_bval_path, get_bvec_path, get_json_path]:
        src_file = Path(get_path(src))
        if src_file.exists():
            dst_file = Path(get_path(dst))
            shutil.copy2(src_file, dst_file)


def find_dwi_files(input_dir: Path) -> list:
    """Find all NIfTI files in a directory."""
    files = sorted(input_dir.glob("*.nii.gz")) + sorted(input_dir.glob("*.nii"))
    # Remove duplicates (file.nii when file.nii.gz also exists)
    gz_stems = {
        f.name.replace(".nii.gz", "")
        for f in files if f.name.endswith(".nii.gz")
    }
    files = [
        f for f in files
        if not (f.name.endswith(".nii") and f.stem in gz_stems)
    ]
    return files


def downsample_folder(input_dir: Path, output_dir: Path, factors: list,
                      nthreads: int = None):
    """Downsample all DWI files in a directory at multiple factors.

    Creates subdirectories under output_dir for each factor:
        output_dir/downsampled_0p5/raw/
        output_dir/downsampled_0p6/raw/
        ...
    """
    dwi_files = find_dwi_files(input_dir)
    if not dwi_files:
        raise ValueError(f"No NIfTI files found in {input_dir}")

    print(f"Found {len(dwi_files)} DWI file(s) in {input_dir}")
    print(f"Downsampling factors: {factors}")
    print()

    for factor in factors:
        factor_str = f"{factor:.1f}".replace(".", "p")
        raw_dir = output_dir / f"downsampled_{factor_str}" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- Factor {factor:.2f} (voxel scale: {1.0/factor:.2f}x) ---")

        for i, dwi in enumerate(dwi_files, 1):
            output_file = raw_dir / dwi.name
            print(f"  [{i}/{len(dwi_files)}] {dwi.name}")

            downsample_dwi(dwi, output_file, factor, nthreads=nthreads)
            copy_associated_files(dwi, output_file)

        print(f"  Saved to: {raw_dir}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Downsample denoised DWI data with sinc interpolation "
            "using MRtrix mrgrid."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing denoised DWI NIfTI files."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Base output directory. Subdirectories created per factor."
    )
    parser.add_argument(
        "--factors", type=float, nargs="+", default=DOWNSAMPLE_FACTORS,
        help=f"Resolution factors (0-1). Default: {DOWNSAMPLE_FACTORS}"
    )
    parser.add_argument(
        "--nthreads", type=int, default=None,
        help="Number of threads for MRtrix."
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    downsample_folder(
        args.input_dir, args.output_dir, args.factors, nthreads=args.nthreads
    )

    print("Done.")


if __name__ == "__main__":
    main()
