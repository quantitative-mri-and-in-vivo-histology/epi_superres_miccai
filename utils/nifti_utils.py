"""NIfTI file utilities."""

import json
from pathlib import Path


def strip_nifti_ext(path: Path) -> Path:
    """Remove .nii or .nii.gz extension from a path.

    Parameters
    ----------
    path : Path
        Path to NIfTI file

    Returns
    -------
    Path
        Path without NIfTI extension
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7])
    elif name.endswith(".nii"):
        return path.with_name(name[:-4])
    return path


def get_bval_path(nifti_path: Path) -> Path:
    """Get the bval file path for a NIfTI file.

    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file (.nii or .nii.gz)

    Returns
    -------
    Path
        Path to corresponding .bval file
    """
    return strip_nifti_ext(nifti_path).with_suffix(".bval")


def get_bvec_path(nifti_path: Path) -> Path:
    """Get the bvec file path for a NIfTI file.

    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file (.nii or .nii.gz)

    Returns
    -------
    Path
        Path to corresponding .bvec file
    """
    return strip_nifti_ext(nifti_path).with_suffix(".bvec")


def with_nifti_ext(path: Path, ext: str) -> Path:
    """Replace NIfTI extension or append a new extension.

    Parameters
    ----------
    path : Path
        Path (with or without NIfTI extension)
    ext : str
        New extension (e.g., ".bval", ".json", ".nii.gz")

    Returns
    -------
    Path
        Path with new extension
    """
    base = strip_nifti_ext(path)
    if not ext.startswith("."):
        ext = "." + ext
    return base.with_suffix(ext)


def get_json_path(nifti_path: Path) -> Path:
    """Get the JSON sidecar file path for a NIfTI file.

    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file (.nii or .nii.gz)

    Returns
    -------
    Path
        Path to corresponding .json file
    """
    return strip_nifti_ext(nifti_path).with_suffix(".json")


def load_json_sidecar(nifti_path: Path) -> dict | None:
    """Load JSON sidecar for a NIfTI file.

    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file

    Returns
    -------
    dict or None
        JSON contents, or None if file doesn't exist
    """
    json_path = get_json_path(nifti_path)
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def get_readout_time(nifti_path: Path) -> float | None:
    """Get total readout time from JSON sidecar.

    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file

    Returns
    -------
    float or None
        TotalReadoutTime in seconds, or None if not found
    """
    data = load_json_sidecar(nifti_path)
    if data is not None:
        return data.get("TotalReadoutTime")
    return None


# BIDS PE direction to MRtrix format mapping
_PE_DIR_MAP = {
    "i": "LR",
    "i-": "RL",
    "j": "PA",
    "j-": "AP",
    "k": "IS",
    "k-": "SI",
}


def get_pe_direction(nifti_path: Path) -> str | None:
    """Get phase encoding direction from JSON sidecar.

    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file

    Returns
    -------
    str or None
        Phase encoding direction in MRtrix format (e.g., "LR", "AP"),
        or None if not found
    """
    data = load_json_sidecar(nifti_path)
    if data is not None:
        pe_dir = data.get("PhaseEncodingDirection")
        if pe_dir is not None:
            return _PE_DIR_MAP.get(pe_dir, pe_dir)
    return None
