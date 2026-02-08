"""Common DWI preprocessing functions.

Reusable building blocks for denoising, sidecar copying, and file utilities
used by dataset-specific scripts.
"""

import shutil
from pathlib import Path

from utils.cmd_utils import run_command
from utils.nifti_utils import get_bval_path, get_bvec_path, get_json_path, strip_nifti_ext


def add_suffix(nii_path: Path, suffix: str) -> Path:
    """Add a suffix before .nii.gz, e.g. _denoised.

    Parameters
    ----------
    nii_path : Path
        Path to NIfTI file
    suffix : str
        Suffix to add (e.g. "_denoised", "_noisemap")

    Returns
    -------
    Path
        Path with suffix inserted before .nii.gz
    """
    base = strip_nifti_ext(nii_path)
    return base.with_name(base.name + suffix + ".nii.gz")


def find_dwi_files(input_dir: Path, pattern: str = "*.nii.gz") -> list[Path]:
    """Find NIfTI files in a directory.

    Parameters
    ----------
    input_dir : Path
        Directory to search
    pattern : str
        Glob pattern (default: "*.nii.gz")

    Returns
    -------
    list[Path]
        Sorted list of matching NIfTI paths
    """
    return sorted(input_dir.glob(pattern))


def copy_sidecar(input_nii: Path, output_nii: Path) -> None:
    """Copy bval, bvec, and json sidecars to match the output NIfTI."""
    for get_path in (get_bval_path, get_bvec_path, get_json_path):
        src = get_path(input_nii)
        dst = get_path(output_nii)
        if src.exists():
            shutil.copy2(src, dst)


def denoise_dwi(
    dwi: Path,
    output: Path,
    noisemap: Path | None = None,
    extent: tuple[int, int, int] | None = None,
    nthreads: int | None = None,
) -> None:
    """Run MRtrix dwidenoise (Marchenko-Pastur PCA).

    Parameters
    ----------
    dwi : Path
        Input DWI data
    output : Path
        Output denoised DWI
    noisemap : Path, optional
        Output noise map
    extent : tuple[int, int, int], optional
        Sliding window extent (default: auto, typically 5x5x5)
    nthreads : int, optional
        Number of threads (0=all available)
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = f"dwidenoise {dwi} {output} -force"

    if noisemap is not None:
        cmd += f" -noise {noisemap}"

    if extent is not None:
        cmd += f" -extent {extent[0]},{extent[1]},{extent[2]}"

    if nthreads is not None:
        cmd += f" -nthreads {nthreads}"

    run_command(cmd)
