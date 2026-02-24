#!/usr/bin/env python3
"""Extract brain masks from MTsat using ANTs brain extraction.

Uses the OMM-1 T1 whole-head template with antsBrainExtraction.sh.
Processes all subjects in single_pe_rpe dataset.

Input:  data/single_pe_rpe/native_res/processed/sub-*/mpm/sub-*_MTsat.nii
Output: data/single_pe_rpe/native_res/processed/sub-*/mpm/mtsat_brain_mask.nii.gz
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cmd_utils import run_command

PROCESSED_BASE = ROOT / "data/single_pe_rpe/native_res/processed"
TEMPLATE = ROOT / "data/templates/OMM-1_T1_head.nii.gz"
TEMPLATE_MASK = ROOT / "data/templates/OMM-1_T1_brain_mask_average.nii.gz"


def mask_mtsat_subject(subject_dir: Path) -> Path | None:
    """Extract brain mask for one subject.

    Parameters
    ----------
    subject_dir : Path
        Subject directory in processed (e.g., .../sub-V06460)

    Returns
    -------
    Path | None
        Brain mask path if successful, None otherwise
    """
    subject_id = subject_dir.name

    # Find MTsat file in mpm directory
    mpm_dir = subject_dir / "mpm"
    if not mpm_dir.exists():
        print(f"  No mpm directory for {subject_id}")
        return None

    mtsat_files = list(mpm_dir.glob(f"{subject_id}_MTsat.nii"))
    if not mtsat_files:
        print(f"  No MTsat file found for {subject_id}")
        return None

    mtsat = mtsat_files[0]
    print(f"  Input: {mtsat.name}")

    # Output directory is same as input (mpm directory)
    output_mpm_dir = mpm_dir

    tmp_dir = output_mpm_dir / "_mask_tmp_mtsat"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prefix = tmp_dir / "ants_"
    print(f"  Running antsBrainExtraction.sh...")
    run_command(
        [
            "antsBrainExtraction.sh",
            "-d", "3",
            "-a", str(mtsat),
            "-e", str(TEMPLATE),
            "-m", str(TEMPLATE_MASK),
            "-o", str(prefix),
        ],
        verbose=False,
    )

    brain_mask = output_mpm_dir / "mtsat_brain_mask.nii.gz"
    Path(f"{prefix}BrainExtractionMask.nii.gz").rename(brain_mask)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  Output: {brain_mask}")
    return brain_mask


def main():
    """Process all subjects in single_pe_rpe dataset."""
    print("Extracting brain masks from MTsat data")
    print("=" * 60)

    if not PROCESSED_BASE.exists():
        print(f"Processed directory not found: {PROCESSED_BASE}")
        return

    # Find all subject directories
    subjects = sorted([d for d in PROCESSED_BASE.iterdir()
                      if d.is_dir() and d.name.startswith("sub-")])

    print(f"Found {len(subjects)} subject(s)")
    print()

    for subject_dir in subjects:
        print(f"=== {subject_dir.name} ===")
        try:
            mask_mtsat_subject(subject_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
