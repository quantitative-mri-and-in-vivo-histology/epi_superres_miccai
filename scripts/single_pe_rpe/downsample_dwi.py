#!/usr/bin/env python3
"""Downsample DWI data from 1.6mm native to 2.5mm in-plane resolution.

Processes all subjects in single_pe_rpe/native_res/raw/ and outputs to
single_pe_rpe/native_res/processed/ with downsampled DWI volumes and
associated bval, bvec, and JSON files.
"""

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data/single_pe_rpe/native_res/processed"
OUTPUT_DIR = ROOT / "data/single_pe_rpe/downsampled_2p5mm/processed"

NATIVE_RES = 1.6  # mm (slice direction, unchanged)
TARGET_RES = 2.5  # mm (in-plane, downsampled)


def downsample_dwi(input_dwi: Path, output_dwi: Path) -> None:
    """Downsample DWI using MRtrix mrgrid with sinc interpolation.

    Parameters
    ----------
    input_dwi : Path
        Input DWI NIfTI file.
    output_dwi : Path
        Output downsampled DWI file.
    """
    output_dwi.parent.mkdir(parents=True, exist_ok=True)

    # Use -voxel to specify exact voxel size in mm (not -scale which scales matrix)
    # Format: X,Y,Z where Y is slice direction (kept at native res)
    cmd = [
        "mrgrid", str(input_dwi), "regrid", str(output_dwi),
        "-voxel", f"{TARGET_RES},{NATIVE_RES},{TARGET_RES}",
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


def copy_sidecar_files(src_dwi: Path, dst_dwi: Path) -> None:
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
        update_json_resolution(dst_json, NATIVE_RES, TARGET_RES)


def process_subject(subject_dir: Path) -> None:
    """Process all DWI files for one subject.

    Parameters
    ----------
    subject_dir : Path
        Subject directory in raw/ (e.g., raw/sub-V06460).
    """
    subject_id = subject_dir.name
    dwi_dir = subject_dir / "dwi"

    if not dwi_dir.is_dir():
        print(f"  No dwi/ folder, skipping")
        return

    # Find all denoised DWI and noisemap NIfTI files
    denoised_files = sorted(dwi_dir.glob("*_denoised.nii.gz"))
    noisemap_files = sorted(dwi_dir.glob("*_noisemap.nii.gz"))
    dwi_files = denoised_files + noisemap_files

    if not dwi_files:
        print(f"  No DWI files found, skipping")
        return

    print(f"  Found {len(denoised_files)} denoised file(s) and {len(noisemap_files)} noisemap(s)")

    # Process each DWI file
    for dwi_file in dwi_files:
        output_dir = OUTPUT_DIR / subject_id / "dwi"
        output_file = output_dir / dwi_file.name

        print(f"    Downsampling {dwi_file.name}")
        downsample_dwi(dwi_file, output_file)
        copy_sidecar_files(dwi_file, output_file)


def main():
    """Process all subjects in the single_pe_rpe dataset."""
    if not RAW_DIR.is_dir():
        raise ValueError(f"Raw directory does not exist: {RAW_DIR}")

    subjects = sorted([d for d in RAW_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("sub-")])

    print(f"Downsampling DWI from {NATIVE_RES}mm to {TARGET_RES}mm in-plane")
    print(f"Found {len(subjects)} subject(s)")
    print()

    for subject_dir in subjects:
        print(f"{subject_dir.name}")
        process_subject(subject_dir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
