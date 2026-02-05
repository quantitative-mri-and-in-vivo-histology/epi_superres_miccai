"""Denoise DWI data using MRtrix dwidenoise.

This script applies Marchenko-Pastur PCA denoising to DWI data in an input
folder, copying associated files (bvec, bval, json) to the output folder.
"""

import argparse
import shutil
from pathlib import Path

from utils.cmd_utils import run_command
from utils.nifti_utils import get_bval_path, get_bvec_path, get_json_path


def run_dwidenoise(
    dwi: Path,
    output: Path,
    extent: tuple[int, int, int] | None = None,
    noise_map: Path | None = None,
    nthreads: int | None = None,
) -> Path:
    """Run MRtrix dwidenoise for Marchenko-Pastur PCA denoising.

    Parameters
    ----------
    dwi : Path
        Input DWI data
    output : Path
        Output denoised DWI
    extent : tuple[int, int, int], optional
        Sliding window extent (default: auto, typically 5x5x5)
    noise_map : Path, optional
        Output noise map if desired
    nthreads : int, optional
        Number of threads (0=all available)

    Returns
    -------
    Path
        Path to denoised DWI
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = f"dwidenoise {dwi} {output} -force"

    if extent is not None:
        cmd += f" -extent {extent[0]},{extent[1]},{extent[2]}"

    if noise_map is not None:
        cmd += f" -noise {noise_map}"

    if nthreads is not None:
        cmd += f" -nthreads {nthreads}"

    run_command(cmd)
    return output


def copy_associated_files(input_nifti: Path, output_nifti: Path) -> list[Path]:
    """Copy bvec, bval, and json files associated with a NIfTI file.

    Parameters
    ----------
    input_nifti : Path
        Input NIfTI file path
    output_nifti : Path
        Output NIfTI file path

    Returns
    -------
    list[Path]
        List of copied file paths
    """
    copied = []

    # Define associated file getters
    file_getters = [get_bvec_path, get_bval_path, get_json_path]

    for get_path in file_getters:
        input_path = get_path(input_nifti)
        output_path = get_path(output_nifti)

        if input_path.exists():
            shutil.copy2(input_path, output_path)
            copied.append(output_path)
            print(f"  Copied: {input_path.name} -> {output_path}")

    return copied


def find_dwi_files(input_dir: Path) -> list[Path]:
    """Find all NIfTI files in a directory.

    Parameters
    ----------
    input_dir : Path
        Input directory to search

    Returns
    -------
    list[Path]
        List of NIfTI file paths (sorted)
    """
    nifti_files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))
    return sorted(nifti_files)


def denoise_folder(
    input_dir: Path,
    output_dir: Path,
    extent: tuple[int, int, int] | None = None,
    save_noise_map: bool = False,
    nthreads: int | None = None,
) -> list[Path]:
    """Denoise all DWI files in a folder.

    Parameters
    ----------
    input_dir : Path
        Input directory containing DWI files
    output_dir : Path
        Output directory for denoised files
    extent : tuple[int, int, int], optional
        Sliding window extent for denoising
    save_noise_map : bool
        Save noise map for each file
    nthreads : int, optional
        Number of threads

    Returns
    -------
    list[Path]
        List of denoised file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dwi_files = find_dwi_files(input_dir)

    if not dwi_files:
        print(f"No NIfTI files found in {input_dir}")
        return []

    print(f"Found {len(dwi_files)} DWI file(s) to denoise")

    denoised_files = []

    for i, dwi in enumerate(dwi_files, 1):
        print(f"\n[{i}/{len(dwi_files)}] Processing: {dwi.name}")

        output_file = output_dir / dwi.name

        # Optional noise map
        noise_map = None
        if save_noise_map:
            noise_map = output_dir / f"{dwi.stem}_noisemap.nii.gz"
            if dwi.name.endswith(".nii.gz"):
                noise_map = output_dir / f"{dwi.name[:-7]}_noisemap.nii.gz"

        # Run denoising
        run_dwidenoise(
            dwi,
            output_file,
            extent=extent,
            noise_map=noise_map,
            nthreads=nthreads,
        )

        # Copy associated files
        copy_associated_files(dwi, output_file)

        denoised_files.append(output_file)

    return denoised_files


def main():
    parser = argparse.ArgumentParser(
        description="Denoise DWI data using MRtrix dwidenoise"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing DWI files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for denoised files",
    )
    parser.add_argument(
        "--extent",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Sliding window extent (default: auto, typically 5x5x5)",
    )
    parser.add_argument(
        "--save-noise-map",
        action="store_true",
        help="Save noise map for each file",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=None,
        help="Number of threads (default: all available)",
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    print(f"Denoising DWI files")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")

    extent = tuple(args.extent) if args.extent else None

    denoised_files = denoise_folder(
        args.input_dir,
        args.output_dir,
        extent=extent,
        save_noise_map=args.save_noise_map,
        nthreads=args.nthreads,
    )

    print(f"\nDone. Denoised {len(denoised_files)} file(s)")


if __name__ == "__main__":
    main()
