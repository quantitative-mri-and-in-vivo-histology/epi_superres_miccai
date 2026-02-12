#!/usr/bin/env python3
"""Downsample DWI data from 1.6mm native to multiple lower resolutions.

Processes all DWI files in multi_pe_rpe/native_res/processed/dwi/ and outputs
to 5 different downsampled directories with varying in-plane resolutions.
"""

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data/multi_pe_rpe/native_res/processed/dwi"

NATIVE_RES = 1.7  # mm (slice direction, unchanged)

# 4 different downsampling levels (in-plane resolution in mm)
# Factor range: 1.18x to 2.0x of native 1.7mm
TARGET_RESOLUTIONS = [2.0, 2.5, 3.0, 3.4]


def get_output_dir(target_res: float) -> Path:
    """Get output directory for a specific target resolution."""
    res_str = str(target_res).replace(".", "p")
    return ROOT / f"data/multi_pe_rpe/downsampled_{res_str}mm/processed/dwi"


def downsample_dwi(input_dwi: Path, output_dwi: Path, target_res: float) -> None:
    """Downsample DWI using MRtrix mrgrid with sinc interpolation.

    Parameters
    ----------
    input_dwi : Path
        Input DWI NIfTI file.
    output_dwi : Path
        Output downsampled DWI file.
    target_res : float
        Target in-plane resolution in mm.
    """
    output_dwi.parent.mkdir(parents=True, exist_ok=True)

    # Use -voxel to specify exact voxel size in mm (not -scale which scales matrix)
    # Format: X,Y,Z where Y is slice direction (kept at native res)
    cmd = [
        "mrgrid", str(input_dwi), "regrid", str(output_dwi),
        "-voxel", f"{target_res},{NATIVE_RES},{target_res}",
        "-interp", "sinc",
        "-force"
    ]

    subprocess.run(cmd, check=True)


def update_json_resolution(json_path: Path, native_res: float, target_res: float) -> None:
    """Update JSON metadata to reflect downsampled resolution.

    Updates matrix dimensions based on the scaling factor.
    Adds metadata indicating this is a downsampled/derived image.
    """
    if not json_path.exists():
        return

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    scale_factor = native_res / target_res

    # Update matrix dimensions if present
    if 'BaseResolution' in metadata:
        metadata['BaseResolution'] = int(metadata['BaseResolution'] * scale_factor)
    if 'AcquisitionMatrixPE' in metadata:
        metadata['AcquisitionMatrixPE'] = int(metadata['AcquisitionMatrixPE'] * scale_factor)
    if 'ReconMatrixPE' in metadata:
        metadata['ReconMatrixPE'] = int(metadata['ReconMatrixPE'] * scale_factor)

    # Add processing metadata
    metadata['ProcessingDescription'] = f'Downsampled from {native_res}mm to {target_res}mm in-plane using MRtrix mrgrid with sinc interpolation'

    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def copy_sidecar_files(src_dwi: Path, dst_dwi: Path, target_res: float) -> None:
    """Copy bval, bvec, and JSON sidecar files alongside output DWI."""
    # Remove .nii.gz or .nii extension to get base path
    src_base = str(src_dwi).replace(".nii.gz", "").replace(".nii", "")
    dst_base = str(dst_dwi).replace(".nii.gz", "").replace(".nii", "")

    for ext in [".bval", ".bvec"]:
        src_file = Path(f"{src_base}{ext}")
        if src_file.exists():
            dst_file = Path(f"{dst_base}{ext}")
            shutil.copy2(src_file, dst_file)

    # Copy and update JSON
    src_json = Path(f"{src_base}.json")
    if src_json.exists():
        dst_json = Path(f"{dst_base}.json")
        shutil.copy2(src_json, dst_json)
        update_json_resolution(dst_json, NATIVE_RES, target_res)


def process_dwi_file(dwi_file: Path, target_res: float) -> None:
    """Process one DWI file for a specific target resolution.

    Parameters
    ----------
    dwi_file : Path
        Input DWI file path.
    target_res : float
        Target in-plane resolution in mm.
    """
    output_dir = get_output_dir(target_res)
    output_file = output_dir / dwi_file.name

    print(f"    {target_res}mm: {dwi_file.name}")
    downsample_dwi(dwi_file, output_file, target_res)
    copy_sidecar_files(dwi_file, output_file, target_res)


def main():
    """Process all DWI files in the multi_pe_rpe dataset."""
    if not RAW_DIR.is_dir():
        raise ValueError(f"Raw directory does not exist: {RAW_DIR}")

    # Find all denoised DWI and noisemap NIfTI files
    denoised_files = sorted(RAW_DIR.glob("*_denoised.nii.gz"))
    noisemap_files = sorted(RAW_DIR.glob("*_noisemap.nii.gz"))
    dwi_files = denoised_files + noisemap_files

    if not dwi_files:
        print("No DWI files found")
        return

    print(f"Downsampling from {NATIVE_RES}mm native resolution")
    print(f"Target resolutions: {TARGET_RESOLUTIONS}")
    print(f"Found {len(denoised_files)} denoised file(s) and {len(noisemap_files)} noisemap(s)")
    print()

    for dwi_file in dwi_files:
        print(f"  {dwi_file.name}")
        for target_res in TARGET_RESOLUTIONS:
            process_dwi_file(dwi_file, target_res)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
