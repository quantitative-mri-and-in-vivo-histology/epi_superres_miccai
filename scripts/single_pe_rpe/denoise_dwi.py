"""Denoise DWI data for single_pe_rpe dataset.

Reads merged DWI files from raw/sub-*/dwi/, runs MRtrix dwidenoise,
and writes denoised + noisemap outputs to processed/sub-*/dwi/.
"""

from pathlib import Path

from scripts.common.preprocessing import add_suffix, copy_sidecar, denoise_dwi, find_dwi_files

BASE_DIR = Path("data/single_pe_rpe/native_res")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"


def main():
    subjects = sorted(
        d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith("sub-")
    )
    print(f"Found {len(subjects)} subject(s)")

    for sub_dir in subjects:
        dwi_dir = sub_dir / "dwi"
        if not dwi_dir.is_dir():
            print(f"\n{sub_dir.name}: no dwi/ folder, skipping")
            continue

        out_dir = PROCESSED_DIR / sub_dir.name / "dwi"
        out_dir.mkdir(parents=True, exist_ok=True)

        dwi_files = find_dwi_files(dwi_dir)
        print(f"\n{sub_dir.name}: {len(dwi_files)} DWI file(s)")

        for dwi in dwi_files:
            out_nii = add_suffix(out_dir / dwi.name, "_denoised")
            out_noise = add_suffix(out_dir / dwi.name, "_noisemap")

            print(f"  {dwi.name} -> {out_nii.name}")
            denoise_dwi(dwi, out_nii, noisemap=out_noise)
            copy_sidecar(dwi, out_nii)

    print("\nDone.")


if __name__ == "__main__":
    main()
