#!/usr/bin/env python3
"""Extract brain mask from MPRAGE using ANTs brain extraction.

Uses the OMM-1 T1 whole-head template with antsBrainExtraction.sh.
Input:  data/multi_pe_rpe/native_res/processed/anat/mprage.nii.gz
Output: data/multi_pe_rpe/native_res/processed/anat/mprage_brain_mask.nii.gz
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cmd_utils import run_command

ANAT_DIR = ROOT / "data/multi_pe_rpe/native_res/processed/anat"
TEMPLATE = ROOT / "data/templates/OMM-1_T1_head.nii.gz"
TEMPLATE_MASK = ROOT / "data/templates/OMM-1_T1_brain_mask_average.nii.gz"


def mask_mprage() -> Path:
    mprage = ANAT_DIR / "mprage.nii.gz"
    if not mprage.exists():
        raise FileNotFoundError(f"MPRAGE not found: {mprage}")

    tmp_dir = ANAT_DIR / "_mask_tmp_mprage"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    prefix = tmp_dir / "ants_"
    print(f"Running antsBrainExtraction.sh on {mprage.name}...")
    run_command(
        [
            "antsBrainExtraction.sh",
            "-d", "3",
            "-a", str(mprage),
            "-e", str(TEMPLATE),
            "-m", str(TEMPLATE_MASK),
            "-o", str(prefix),
        ],
        verbose=True,
    )

    brain_mask = ANAT_DIR / "mprage_brain_mask.nii.gz"
    Path(f"{prefix}BrainExtractionMask.nii.gz").rename(brain_mask)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Brain mask: {brain_mask}")
    return brain_mask


def main():
    mask_mprage()
    print("Done.")


if __name__ == "__main__":
    main()
