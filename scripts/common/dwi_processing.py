"""Common DWI processing functions shared across datasets.

This module contains reusable functions for DWI preprocessing, masking,
tensor fitting, and registration workflows.
"""

import shutil
from pathlib import Path

from scripts.common.preprocessing import (
    create_brain_mask,
    fit_tensors,
    register_to_anat_ref,
    run_dwifslpreproc,
)
from utils.cmd_utils import run_command


def process_single_mode(
    dwi_forward: Path,
    dwi_reverse: Path,
    dwi_dir: Path,
    mode: str,
    nthreads: int = 0,
    anat_ref_image: Path | None = None,
    anat_mask_image: Path | None = None,
    skip_preproc: bool = False,
    preprocessed_path: Path | None = None,
    eddy_dir_path: Path | None = None,
) -> None:
    """Process DWI pair with a specific mode (b0_rpe or full_rpe).

    Parameters
    ----------
    dwi_forward : Path
        Forward phase encoding denoised DWI
    dwi_reverse : Path
        Reverse phase encoding denoised DWI
    dwi_dir : Path
        DWI directory (for output)
    mode : str
        Processing mode: "b0_rpe" or "full_rpe"
    nthreads : int
        Number of threads for topup
    anat_ref_image : Path, optional
        Anatomical reference image for registration
    anat_mask_image : Path, optional
        Anatomical brain mask to regrid and use (instead of creating from DWI)
    skip_preproc : bool
        Skip dwifslpreproc, use existing preprocessed DWI
    preprocessed_path : Path, optional
        Expected path to preprocessed DWI (required if skip_preproc=True)
    eddy_dir_path : Path, optional
        Expected path to eddy output directory (required if skip_preproc=True)
    """
    print(f"  === Mode: {mode} ===")

    if skip_preproc:
        if preprocessed_path is None or eddy_dir_path is None:
            raise ValueError("preprocessed_path and eddy_dir_path required when skip_preproc=True")
        if not preprocessed_path.exists():
            raise FileNotFoundError(
                f"--skip-preproc: expected preprocessed DWI at {preprocessed_path}"
            )
        preprocessed = preprocessed_path
        eddy_dir = eddy_dir_path
        print(f"  Skipping preprocessing, using existing: {preprocessed.name}")
    else:
        # Step 1: Preprocessing (dwifslpreproc)
        print(f"  Step 1: Preprocessing (topup/eddy)...")
        preprocessed, eddy_dir = run_dwifslpreproc(
            dwi_forward, dwi_reverse, dwi_dir, mode=mode, nthreads=nthreads
        )
        print(f"    Output: {preprocessed.name}")

    # Step 2: Brain masking
    print(f"  Step 2: Brain masking...")
    prefix = preprocessed.stem.replace(".nii", "")
    final_mask = dwi_dir / f"{prefix}_brain_mask.nii.gz"

    if anat_mask_image is not None and anat_mask_image.exists():
        # Regrid and reorient anatomical mask to match preprocessed DWI exactly
        print(f"    Regridding anatomical mask: {anat_mask_image.name}")

        # Transform mask to match preprocessed DWI template (geometry + strides)
        # Using -template alone doesn't guarantee stride matching, so we also
        # explicitly copy the DWI strides to ensure nibabel loads them identically
        mask_tmp = dwi_dir / f"_tmp_mask_{mode}.nii.gz"
        run_command(
            f"mrtransform {anat_mask_image} -template {preprocessed} "
            f"-interp nearest {mask_tmp} -force",
            verbose=False,
        )

        # Force mask to have identical strides as preprocessed DWI
        run_command(
            f"mrconvert {mask_tmp} -strides {preprocessed} {final_mask} -force",
            verbose=False,
        )

        # Clean up
        mask_tmp.unlink(missing_ok=True)

        print(f"    Brain mask (from anat): {final_mask.name}")
    else:
        # Create brain mask from DWI
        mask_tmp = dwi_dir / f"_mask_tmp_{mode}"
        brain_mask = create_brain_mask(preprocessed, mask_tmp)
        brain_mask.rename(final_mask)
        shutil.rmtree(mask_tmp, ignore_errors=True)
        print(f"    Brain mask (from DWI): {final_mask.name}")

    # Create masked DWI
    masked_dwi = dwi_dir / f"{prefix}_masked.nii.gz"
    run_command(f"mrcalc {preprocessed} {final_mask} -mult {masked_dwi} -force", verbose=False)
    print(f"    Masked DWI: {masked_dwi.name}")

    # Step 3: Tensor fitting
    print(f"  Step 3: Tensor fitting...")
    output_prefix = dwi_dir / prefix
    fit_tensors(preprocessed, output_prefix, final_mask)

    # Step 4: Anatomical registration (optional)
    if anat_ref_image is not None and anat_ref_image.exists():
        print(f"  Step 4: Anatomical registration...")

        dwi_anat, anat_mask = register_to_anat_ref(
            preprocessed_dwi=preprocessed,
            eddy_output_dir=eddy_dir,
            anat_ref_image=anat_ref_image,
            output_dir=dwi_dir,
            anat_mask_image=anat_mask_image,
            dwi_mask=final_mask,
            nthreads=nthreads,
        )

        print(f"  Step 5: Tensor fitting on registered data...")
        anat_output_prefix = dwi_dir / dwi_anat.stem.replace(".nii", "")
        fit_tensors(dwi_anat, anat_output_prefix, anat_mask)

        # Step 6: Tissue segmentation using multi-channel Atropos (FA + S0)
        print(f"  Step 6: Tissue segmentation (ANTs Atropos, multi-channel)...")
        anat_fa = Path(f"{anat_output_prefix}_dti_fa.nii.gz")
        anat_s0 = Path(f"{anat_output_prefix}_dti_s0.nii.gz")

        if anat_fa.exists() and anat_s0.exists():
            seg_path = dwi_dir / f"{anat_output_prefix.name}_segmentation.nii.gz"
            prob_prefix = str(dwi_dir / f"{anat_output_prefix.name}_segmentation_prob")

            # Multi-channel Atropos: combine FA (tissue contrast) + S0 (T2-weighted contrast)
            # Use multiple -a flags for multi-channel input
            cmd = [
                "Atropos",
                "-d", "3",
                "-a", str(anat_fa),              # Channel 1: FA (tissue contrast)
                "-a", str(anat_s0),              # Channel 2: S0 (T2-weighted contrast)
                "-x", str(anat_mask),
                "-i", "KMeans[3]",               # 3-tissue k-means initialization
                "-c", "[5,0]",                   # 5 iterations, no partial volume
                "-m", "[0.1,1x1x1]",             # MRF smoothing (weight=0.1, radius=1)
                "-o", f"[{seg_path},{prob_prefix}_%02d.nii.gz]"
            ]
            run_command(cmd, verbose=False)
            print(f"    Segmentation: {seg_path.name}")

            # Rename probability maps to meaningful names (1=CSF, 2=GM, 3=WM)
            prob_csf = dwi_dir / f"{anat_output_prefix.name}_segmentation_prob_csf.nii.gz"
            prob_gm = dwi_dir / f"{anat_output_prefix.name}_segmentation_prob_gm.nii.gz"
            prob_wm = dwi_dir / f"{anat_output_prefix.name}_segmentation_prob_wm.nii.gz"

            Path(f"{prob_prefix}_01.nii.gz").rename(prob_csf)
            Path(f"{prob_prefix}_02.nii.gz").rename(prob_gm)
            Path(f"{prob_prefix}_03.nii.gz").rename(prob_wm)

            print(f"    Probability maps: {prob_csf.name}, {prob_gm.name}, {prob_wm.name}")
        else:
            print(f"    Skipping segmentation: FA or S0 not found")

    print(f"  ✓ Completed {mode} mode")
    print()
