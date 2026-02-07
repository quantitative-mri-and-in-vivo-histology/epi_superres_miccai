"""Process phase encoding data using MRtrix dwifslpreproc.

This script runs the standard FSL distortion correction pipeline via
MRtrix's dwifslpreproc wrapper on DKI data with reverse phase encoding pairs.

It saves eddy's per-volume displacement fields and outlier-free data,
enabling later single-interpolation warp combination (e.g., combining
eddy correction with DWI-to-T1w registration in one step).
"""

import argparse
import glob
import os
import shutil
import tempfile
from pathlib import Path

from utils.cmd_utils import run_command
from utils.nifti_utils import get_bval_path, get_bvec_path, get_json_path, get_pe_direction, get_readout_time
from utils.warp_utils import combine_and_apply_warps, convert_ants_to_flirt


def extract_b0(dwi: Path, output: Path, bvec: Path | None = None, bval: Path | None = None) -> Path:
    """Extract b0 volumes from DWI.

    Parameters
    ----------
    dwi : Path
        Input DWI data
    output : Path
        Output b0 image
    bvec : Path, optional
        b-vectors file (auto-detected if None)
    bval : Path, optional
        b-values file (auto-detected if None)

    Returns
    -------
    Path
        Path to b0 image
    """
    if bvec is None:
        bvec = get_bvec_path(dwi)
    if bval is None:
        bval = get_bval_path(dwi)

    cmd = f"dwiextract -bzero -force -fslgrad {bvec} {bval} {dwi} {output}"
    run_command(cmd)
    return output


def run_dwifslpreproc(
    dwi: Path,
    dwi_rpe: Path,
    output_dir: Path,
    readout_time: float | None = None,
    nthreads: int = 0,
    keep_tmp: bool = False,
) -> tuple[Path, Path]:
    """Run MRtrix dwifslpreproc for distortion/motion correction.

    Saves eddy's per-volume displacement fields and outlier-free data
    to ``output_dir/eddy_output/``, enabling later single-interpolation
    warp combination.

    Parameters
    ----------
    dwi : Path
        Input DWI data (primary PE direction)
    dwi_rpe : Path
        DWI data with reverse PE direction (for topup)
    output_dir : Path
        Output directory (filename preserved from input)
    readout_time : float, optional
        Total readout time in seconds (read from JSON if not provided)
    nthreads : int
        Number of threads for topup (0=all available)
    keep_tmp : bool
        Keep temporary directory for debugging

    Returns
    -------
    tuple[Path, Path]
        (corrected DWI path, eddy output directory)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / dwi.name

    # Get bval/bvec/json files
    bvec = get_bvec_path(dwi)
    bval = get_bval_path(dwi)
    bvec_rpe = get_bvec_path(dwi_rpe)
    bval_rpe = get_bval_path(dwi_rpe)
    json_file = get_json_path(dwi)

    # Get PE direction from JSON
    pe_dir = get_pe_direction(dwi)
    if pe_dir is None:
        raise ValueError(
            "PE direction not found in JSON sidecar. "
            "Ensure PhaseEncodingDirection is in the JSON file."
        )
    print(f"Using PE direction from JSON: {pe_dir}")

    # Get readout time from JSON if not provided
    if readout_time is None:
        readout_time = get_readout_time(dwi)
        if readout_time is not None:
            print(f"Using readout time from JSON: {readout_time}")

    if readout_time is None:
        raise ValueError(
            "Readout time not specified and not found in JSON sidecar. "
            "Please provide --readout-time or ensure TotalReadoutTime is in the JSON file."
        )

    # Create temp directory
    if keep_tmp:
        tmp_dir = output.parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="dwifslpreproc_"))
    print(f"Temp directory: {tmp_dir}")

    # Eddy output directory (persisted after processing)
    eddy_output_dir = output_dir / "eddy_output"
    eddy_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract first b0 from both PE directions
        b0_pe = tmp_dir / "b0_pe.nii.gz"
        b0_rpe = tmp_dir / "b0_rpe.nii.gz"
        run_command(f"mrconvert {dwi} {b0_pe} -force -coord 3 0 -fslgrad {bvec} {bval}")
        run_command(f"mrconvert {dwi_rpe} {b0_rpe} -force -coord 3 0 -fslgrad {bvec_rpe} {bval_rpe}")

        # Merge b0s for topup (PE + RPE)
        b0_pair = tmp_dir / "b0_pair.nii.gz"
        run_command(f"mrcat {b0_pe} {b0_rpe} {b0_pair} -force -axis 3")

        # Output bvec/bval paths
        out_bvec = get_bvec_path(output)
        out_bval = get_bval_path(output)

        # Scratch path for dwifslpreproc (must NOT exist — MRtrix creates it)
        scratch_dir = tmp_dir / "dwifslpreproc"

        # Run dwifslpreproc with -nocleanup to preserve scratch for dfields,
        # and -eddyqc_all to persist eddy QC outputs (incl. outlier-free data)
        cmd = (
            f"dwifslpreproc {dwi} {output} -force "
            f"-nocleanup "
            f"-fslgrad {bvec} {bval} "
            f"-export_grad_fsl {out_bvec} {out_bval} "
            f"-rpe_pair -se_epi {b0_pair} -pe_dir {pe_dir} "
            f"-readout_time {readout_time} "
            f"-scratch {scratch_dir} "
            f"-eddyqc_all {eddy_output_dir}"
        )

        cmd += f" -topup_options ' --nthr={nthreads}'"
        cmd += f" -eddy_options ' --repol --data_is_shelled --slm=linear --dfields --nthr={nthreads}'"

        run_command(cmd)

        # Copy dfields + dwi_post_eddy from scratch (not saved by -eddyqc_all)
        _extract_eddy_outputs(scratch_dir, eddy_output_dir)

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up temp directory")
        else:
            print(f"Kept temp directory: {tmp_dir}")

    return output, eddy_output_dir


def _extract_eddy_outputs(scratch_dir: Path, eddy_output_dir: Path) -> None:
    """Copy dfields and dwi_post_eddy from dwifslpreproc scratch.

    The ``-eddyqc_all`` flag already saves most eddy outputs (including
    ``eddy_outlier_free_data.nii.gz``) to ``eddy_output_dir``. This
    function copies the remaining files that ``-eddyqc_all`` does not
    save: per-volume displacement fields and ``dwi_post_eddy.nii.gz``.

    Parameters
    ----------
    scratch_dir : Path
        The scratch directory passed to ``-scratch`` (MRtrix uses this
        path directly when it doesn't already exist)
    eddy_output_dir : Path
        Destination directory (already populated by ``-eddyqc_all``)
    """
    # Copy per-volume displacement fields
    dfields_dir = eddy_output_dir / "dfields"
    dfields_dir.mkdir(parents=True, exist_ok=True)

    dfield_files = sorted(glob.glob(
        str(scratch_dir / "dwi_post_eddy.eddy_displacement_fields.*.nii.gz")
    ))
    if not dfield_files:
        print("Warning: No displacement field files found in scratch directory")
    for f in dfield_files:
        shutil.copy(f, dfields_dir)
    print(f"Copied {len(dfield_files)} displacement field(s) to {dfields_dir}")

    # Copy fully corrected dwi_post_eddy (for reference/first-volume extraction)
    dwi_post_eddy = scratch_dir / "dwi_post_eddy.nii.gz"
    if dwi_post_eddy.exists():
        shutil.copy(dwi_post_eddy, eddy_output_dir / "dwi_post_eddy.nii.gz")
        print(f"Saved dwi_post_eddy.nii.gz")


def run_dwi2tensor(
    dwi: Path,
    dt_output: Path,
    dkt_output: Path | None = None,
    b0_output: Path | None = None,
    mask: Path | None = None,
) -> tuple[Path, Path | None, Path | None]:
    """Fit diffusion tensor to DWI data.

    Parameters
    ----------
    dwi : Path
        Input DWI data (preprocessed)
    dt_output : Path
        Output path for diffusion tensor
    dkt_output : Path, optional
        Output path for kurtosis tensor (enables -dkt option if provided)
    b0_output : Path, optional
        Output path for estimated S0 signal
    mask : Path, optional
        Brain mask image

    Returns
    -------
    tuple[Path, Path | None, Path | None]
        Paths to (diffusion tensor, kurtosis tensor or None, S0 or None)
    """
    dt_output.parent.mkdir(parents=True, exist_ok=True)

    # Get bval/bvec files
    bvec = get_bvec_path(dwi)
    bval = get_bval_path(dwi)

    cmd = f"dwi2tensor {dwi} {dt_output} -force -fslgrad {bvec} {bval} -iter 10"

    if mask is not None:
        cmd += f" -mask {mask}"

    if dkt_output is not None:
        cmd += f" -dkt {dkt_output}"

    if b0_output is not None:
        cmd += f" -b0 {b0_output}"

    run_command(cmd)

    return dt_output, dkt_output, b0_output


def run_tensor2metric(
    dt_input: Path,
    fa_output: Path,
    md_output: Path,
) -> tuple[Path, Path]:
    """Compute tensor metrics (FA, MD) from diffusion tensor.

    Parameters
    ----------
    dt_input : Path
        Input diffusion tensor image
    fa_output : Path
        Output path for FA image
    md_output : Path
        Output path for MD image

    Returns
    -------
    tuple[Path, Path]
        Paths to (FA, MD)
    """
    fa_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = f"tensor2metric {dt_input} -force -fa {fa_output} -adc {md_output}"
    run_command(cmd)

    return fa_output, md_output


def _create_brain_mask(dwi: Path, output_dir: Path) -> Path:
    """Create brain mask from DWI using BET on mean b=0.

    Parameters
    ----------
    dwi : Path
        Input DWI (with matching .bvec/.bval)
    output_dir : Path
        Output directory for mask and intermediate files

    Returns
    -------
    Path
        Path to brain mask
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bvec = get_bvec_path(dwi)
    bval = get_bval_path(dwi)

    mean_b0 = output_dir / "mean_b0.nii.gz"
    run_command(
        f"dwiextract -bzero -force -fslgrad {bvec} {bval} {dwi} - "
        f"| mrmath - mean -axis 3 {mean_b0} -force"
    )
    bet_output = output_dir / "mean_b0_brain"
    run_command(f"bet {mean_b0} {bet_output} -m -R")
    brain_mask = output_dir / "brain_mask.nii.gz"
    Path(f"{bet_output}_mask.nii.gz").rename(brain_mask)
    Path(f"{bet_output}.nii.gz").unlink(missing_ok=True)

    # Erode by 1 voxel to tighten the mask
    run_command(f"maskfilter {brain_mask} erode {brain_mask} -force")
    print(f"  Brain mask (eroded): {brain_mask}")

    return brain_mask


def _diag_validity_mask(image: Path, mask: Path, diag_low: float, diag_high: float, tmp_dir: Path) -> Path:
    """Create a validity mask from diagonal tensor elements.

    Returns a 3D binary mask where all diagonal elements (volumes 0-2)
    are within [diag_low, diag_high] and the voxel is inside the brain mask.
    Also replaces NaN/Inf with 0 in the image (in-place).
    """
    stem = image.stem.replace(".nii", "")
    # Replace NaN/Inf with 0
    run_command(f"mrcalc {image} -finite {image} 0 -if {image} -force")
    # Extract diagonal volumes (0, 1, 2)
    diag = tmp_dir / f"_diag_{stem}.nii.gz"
    run_command(f"mrconvert {image} -coord 3 0:2 {diag} -force")
    # Per-voxel validity: all diags in [low, high]
    valid = tmp_dir / f"_valid_{stem}.nii.gz"
    run_command(
        f"mrcalc {diag} {diag_low} -ge {diag} {diag_high} -le -mult "
        f"{mask} -mult {valid} -force"
    )
    # Collapse across volumes: valid only if ALL diags pass
    valid_mask = tmp_dir / f"_valid_mask_{stem}.nii.gz"
    run_command(f"mrmath {valid} min -axis 3 {valid_mask} -force")
    # Clean up intermediates
    diag.unlink(missing_ok=True)
    valid.unlink(missing_ok=True)
    return valid_mask


def _threshold_metric(image: Path, low: float, high: float | None = None) -> None:
    """Zero out voxels outside [low, high] range (in-place).

    If high is None, only apply lower bound.
    """
    if high is not None:
        run_command(
            f"mrcalc {image} {low} -ge {image} {high} -le -mult {image} -mult {image} -force"
        )
    else:
        run_command(
            f"mrcalc {image} {low} -ge {image} -mult {image} -force"
        )


def _fit_tensors(dwi: Path, output_dir: Path, mask: Path) -> None:
    """Fit DTI and DKI tensors and compute metrics.

    Applies physical bounds filtering after metric computation:
    - FA: [0, 1]
    - MD: [0, 0.003] mm²/s
    - S0: > 0
    - MK: [0, 3]

    Parameters
    ----------
    dwi : Path
        Input DWI (with matching .bvec/.bval)
    output_dir : Path
        Output directory for tensor images and metrics
    mask : Path
        Brain mask image
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # DTI
    print("  Fitting DTI...")
    dti_d = output_dir / "dti_d.nii.gz"
    dti_s0 = output_dir / "dti_s0.nii.gz"
    run_dwi2tensor(dwi, dti_d, b0_output=dti_s0, mask=mask)

    print("  Cleaning diffusion tensor...")
    dt_valid = _diag_validity_mask(dti_d, mask, diag_low=0, diag_high=0.003, tmp_dir=tmp_dir)
    run_command(f"mrcalc {dti_d} {dt_valid} -mult {dti_d} -force")
    run_command(f"mrcalc {dti_s0} {dt_valid} -mult {dti_s0} -force")
    dt_valid.unlink(missing_ok=True)

    dti_fa = output_dir / "dti_fa.nii.gz"
    dti_md = output_dir / "dti_md.nii.gz"
    run_tensor2metric(dti_d, dti_fa, dti_md)

    print("  Filtering DTI metrics...")
    _threshold_metric(dti_fa, 0, 1)
    _threshold_metric(dti_md, 0, 0.003)
    s0_mean = float(run_command(
        f"mrstats {dti_s0} -mask {mask} -output mean", verbose=False,
    ).stdout.strip())
    _threshold_metric(dti_s0, 0, s0_mean * 3)
    print(f"  DTI: {dti_fa}, {dti_md}, {dti_s0}")

    # DKI
    print("  Fitting DKI...")
    dki_d = output_dir / "dki_d.nii.gz"
    dki_k = output_dir / "dki_k.nii.gz"
    dki_s0 = output_dir / "dki_s0.nii.gz"
    run_dwi2tensor(dwi, dki_d, dki_k, dki_s0, mask=mask)

    print("  Cleaning tensors...")
    d_valid = _diag_validity_mask(dki_d, mask, diag_low=0, diag_high=0.003, tmp_dir=tmp_dir)
    k_valid = _diag_validity_mask(dki_k, mask, diag_low=0, diag_high=10, tmp_dir=tmp_dir)
    # Combined mask: voxel must pass both D and K checks
    combined_valid = tmp_dir / "_combined_valid.nii.gz"
    run_command(f"mrcalc {d_valid} {k_valid} -mult {combined_valid} -force")
    d_valid.unlink(missing_ok=True)
    k_valid.unlink(missing_ok=True)
    # Apply combined mask to D, K, and S0
    run_command(f"mrcalc {dki_d} {combined_valid} -mult {dki_d} -force")
    run_command(f"mrcalc {dki_k} {combined_valid} -mult {dki_k} -force")
    run_command(f"mrcalc {dki_s0} {combined_valid} -mult {dki_s0} -force")
    combined_valid.unlink(missing_ok=True)

    dki_fa = output_dir / "dki_fa.nii.gz"
    dki_md = output_dir / "dki_md.nii.gz"
    run_tensor2metric(dki_d, dki_fa, dki_md)

    # Compute mean kurtosis from kurtosis tensor
    dki_mk = output_dir / "dki_mk.nii.gz"
    run_command(f"mrmath {dki_k} mean -axis 3 {dki_mk} -force")

    print("  Filtering DKI metrics...")
    _threshold_metric(dki_fa, 0, 1)
    _threshold_metric(dki_md, 0, 0.003)
    s0_mean = float(run_command(
        f"mrstats {dki_s0} -mask {mask} -output mean", verbose=False,
    ).stdout.strip())
    _threshold_metric(dki_s0, 0, s0_mean * 3)
    _threshold_metric(dki_mk, 0, 3)
    print(f"  DKI: {dki_fa}, {dki_md}, {dki_s0}, {dki_mk}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Process DKI data with MRtrix dwifslpreproc"
    )
    parser.add_argument("--dwi", type=Path, required=True, help="Input DWI data")
    parser.add_argument("--dwi-rpe", type=Path, default=None, help="Reverse PE data for topup")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (filename preserved from input)")
    parser.add_argument("--readout-time", type=float, default=None, help="Total readout time (s)")
    parser.add_argument("--nthreads", type=int, default=os.cpu_count(), help="Number of threads for topup")
    parser.add_argument("--t1w", type=Path, default=None, help="T1w image for registration (enables single-interpolation pipeline)")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temp directory for debugging")
    parser.add_argument("--skip-preproc", action="store_true", help="Skip dwifslpreproc, use existing corrected DWI in output-dir")

    args = parser.parse_args()

    preproc_dir = args.output_dir / "preprocessed"

    # --- Step 1: Preprocessing (eddy correction) ---
    if args.skip_preproc:
        dwi_corrected = preproc_dir / args.dwi.name
        if not dwi_corrected.exists():
            raise FileNotFoundError(
                f"--skip-preproc: expected corrected DWI at {dwi_corrected}"
            )
        eddy_output_dir = preproc_dir / "eddy_output"
        print(f"Skipping preprocessing, using existing: {dwi_corrected}")
    else:
        if args.dwi_rpe is None:
            parser.error("--dwi-rpe is required unless --skip-preproc is set")

        print(f"Processing DWI with dwifslpreproc")
        print(f"Input: {args.dwi}")
        print(f"RPE: {args.dwi_rpe}")
        print(f"Output dir: {preproc_dir}")

        dwi_corrected, eddy_output_dir = run_dwifslpreproc(
            args.dwi,
            args.dwi_rpe,
            preproc_dir,
            args.readout_time,
            args.nthreads,
            args.keep_tmp,
        )

        print(f"Preprocessed: {dwi_corrected}")
        print(f"Eddy outputs: {eddy_output_dir}")

    # --- Step 2: Brain masking + tensor fitting on eddy-corrected data ---
    print("\nBrain masking (BET on mean b=0)...")
    brain_mask = _create_brain_mask(dwi_corrected, preproc_dir)

    dwi_masked = preproc_dir / f"{dwi_corrected.name.replace('.nii.gz', '')}_masked.nii.gz"
    run_command(f"mrcalc {dwi_corrected} {brain_mask} -mult {dwi_masked} -force")
    print(f"Masked DWI: {dwi_masked}")

    print("\nFitting tensors on eddy-corrected data...")
    _fit_tensors(dwi_corrected, preproc_dir, brain_mask)

    # --- Step 3: T1w registration + single-interpolation warp combination ---
    if args.t1w is not None:
        if not args.t1w.exists():
            raise FileNotFoundError(f"T1w image not found: {args.t1w}")

        t1w_dir = args.output_dir / "t1w_registered"
        reg_dir = t1w_dir / "reg"
        reg_dir.mkdir(parents=True, exist_ok=True)

        # Extract first volume of dwi_post_eddy with fslroi to preserve
        # FSL headers (consistent with eddy displacement fields)
        eddy_post_vol0 = reg_dir / "dwi_post_eddy_vol0.nii.gz"
        dwi_post_eddy = eddy_output_dir / "dwi_post_eddy.nii.gz"
        run_command(f"fslroi {dwi_post_eddy} {eddy_post_vol0} 0 1")

        # Register eddy first vol → T1w (rigid, mutual information)
        print(f"\nRegistering DWI → T1w (rigid)...")
        reg_prefix = reg_dir / "b0_to_t1w"
        run_command(
            f"antsRegistration --dimensionality 3 --float 0"
            f" --output [{reg_prefix}_,{reg_prefix}_Warped.nii.gz]"
            f" --interpolation Linear"
            f" --use-histogram-matching 1"
            f" --initial-moving-transform [{args.t1w},{eddy_post_vol0},1]"
            f" --transform Rigid[0.1]"
            f" --metric MI[{args.t1w},{eddy_post_vol0},1,32,Regular,0.25]"
            f" --convergence [1000x500x250x100,1e-6,10]"
            f" --smoothing-sigmas 3x2x1x0vox"
            f" --shrink-factors 8x4x2x1"
        )
        ants_mat = Path(f"{reg_prefix}_0GenericAffine.mat")
        print(f"ANTs transform: {ants_mat}")

        # Convert ANTs transform → FLIRT format
        # Use eddy_post_vol0 for both ref and src to keep FSL headers
        # consistent with the eddy displacement fields in convertwarp
        print("Converting ANTs transform to FLIRT format...")
        flirt_mat = reg_dir / "b0_to_t1w_flirt.txt"
        convert_ants_to_flirt(
            ants_mat, ref=eddy_post_vol0, src=eddy_post_vol0, output=flirt_mat,
        )

        # Combine eddy warps + registration in single interpolation
        # Output is on DWI grid with content aligned to T1w
        print("\nApplying combined warps (eddy + registration)...")
        dwi_t1w = t1w_dir / "dwi_t1w.nii.gz"
        combine_and_apply_warps(
            eddy_output_dir, dwi_t1w,
            postmat=flirt_mat,
        )

        # Copy bvecs/bvals to T1w-space output
        shutil.copy(get_bvec_path(dwi_corrected), get_bvec_path(dwi_t1w))
        shutil.copy(get_bval_path(dwi_corrected), get_bval_path(dwi_t1w))
        print(f"T1w-space DWI: {dwi_t1w}")

        # Brain masking + tensor fitting on T1w-registered data
        print("\nBrain masking (BET on mean b=0)...")
        t1w_brain_mask = _create_brain_mask(dwi_t1w, t1w_dir)

        dwi_t1w_masked = t1w_dir / f"{dwi_t1w.name.replace('.nii.gz', '')}_masked.nii.gz"
        run_command(f"mrcalc {dwi_t1w} {t1w_brain_mask} -mult {dwi_t1w_masked} -force")
        print(f"Masked DWI: {dwi_t1w_masked}")

        print("\nFitting tensors on T1w-registered data...")
        _fit_tensors(dwi_t1w, t1w_dir, t1w_brain_mask)

    print("\nDone")


if __name__ == "__main__":
    main()
