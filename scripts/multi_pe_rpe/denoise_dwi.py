"""Denoise DWI data for multi_pe_rpe dataset.

Reads all DWI files from raw/dwi/, runs MRtrix dwidenoise,
and writes denoised + noisemap outputs to processed/dwi/.
"""

from pathlib import Path

from scripts.common.preprocessing import add_suffix, copy_sidecar, denoise_dwi, find_dwi_files

BASE_DIR = Path("data/multi_pe_rpe/native_res")
RAW_DWI = BASE_DIR / "raw" / "dwi"
PROCESSED_DWI = BASE_DIR / "processed" / "dwi"


def main():
    if not RAW_DWI.is_dir():
        raise ValueError(f"Raw DWI directory does not exist: {RAW_DWI}")

    PROCESSED_DWI.mkdir(parents=True, exist_ok=True)

    dwi_files = find_dwi_files(RAW_DWI, pattern="dwi_*.nii.gz")
    print(f"Found {len(dwi_files)} DWI file(s)")

    for dwi in dwi_files:
        out_nii = add_suffix(PROCESSED_DWI / dwi.name, "_denoised")
        out_noise = add_suffix(PROCESSED_DWI / dwi.name, "_noisemap")

        print(f"  {dwi.name} -> {out_nii.name}")
        denoise_dwi(dwi, out_nii, noisemap=out_noise)
        copy_sidecar(dwi, out_nii)

    print("\nDone.")


if __name__ == "__main__":
    main()
