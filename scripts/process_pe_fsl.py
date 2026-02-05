"""Process phase encoding data using MRtrix dwifslpreproc.

This script runs the standard FSL distortion correction pipeline via
MRtrix's dwifslpreproc wrapper on DKI data with reverse phase encoding pairs.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from utils.cmd_utils import run_command
from utils.nifti_utils import get_bval_path, get_bvec_path, get_json_path, get_pe_direction, get_readout_time


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
) -> Path:
    """Run MRtrix dwifslpreproc for distortion/motion correction.

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
    Path
        Path to corrected DWI
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

        # Create scratch directory for dwifslpreproc
        scratch_dir = tmp_dir / "dwifslpreproc"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # Run dwifslpreproc
        cmd = (
            f"dwifslpreproc {dwi} {output} -force "
            f"-fslgrad {bvec} {bval} "
            f"-export_grad_fsl {out_bvec} {out_bval} "
            f"-rpe_pair -se_epi {b0_pair} -pe_dir {pe_dir} "
            f"-readout_time {readout_time} "
            f"-scratch {scratch_dir}"
        )

        cmd += f" -topup_options ' --nthr={nthreads}'"
        cmd += " -eddy_options ' --repol --data_is_shelled --slm=linear'"

        run_command(cmd)

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up temp directory")
        else:
            print(f"Kept temp directory: {tmp_dir}")

    return output


def run_dwi2tensor(
    dwi: Path,
    dt_output: Path,
    dkt_output: Path | None = None,
    b0_output: Path | None = None,
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

    Returns
    -------
    tuple[Path, Path | None, Path | None]
        Paths to (diffusion tensor, kurtosis tensor or None, S0 or None)
    """
    dt_output.parent.mkdir(parents=True, exist_ok=True)

    # Get bval/bvec files
    bvec = get_bvec_path(dwi)
    bval = get_bval_path(dwi)

    cmd = f"dwi2tensor {dwi} {dt_output} -force -fslgrad {bvec} {bval}"

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


def main():
    parser = argparse.ArgumentParser(
        description="Process DKI data with MRtrix dwifslpreproc"
    )
    parser.add_argument("--dwi", type=Path, required=True, help="Input DWI data")
    parser.add_argument("--dwi-rpe", type=Path, required=True, help="Reverse PE data for topup")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (filename preserved from input)")
    parser.add_argument("--readout-time", type=float, default=None, help="Total readout time (s)")
    parser.add_argument("--nthreads", type=int, default=os.cpu_count(), help="Number of threads for topup")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temp directory for debugging")

    args = parser.parse_args()

    output = args.output_dir / args.dwi.name

    print(f"Processing DWI with dwifslpreproc")
    print(f"Input: {args.dwi}")
    print(f"RPE: {args.dwi_rpe}")
    print(f"Output: {output}")

    dwi_corrected = run_dwifslpreproc(
        args.dwi,
        args.dwi_rpe,
        args.output_dir,
        args.readout_time,
        args.nthreads,
        args.keep_tmp,
    )

    print(f"Preprocessed: {dwi_corrected}")

    # Fit DTI (without kurtosis)
    print("Fitting diffusion tensor (DTI)...")
    dti_d = args.output_dir / "dti_d.nii.gz"
    dti_s0 = args.output_dir / "dti_s0.nii.gz"
    run_dwi2tensor(dwi_corrected, dti_d, b0_output=dti_s0)
    print(f"DTI tensor: {dti_d}")

    # Compute DTI metrics
    print("Computing DTI metrics...")
    dti_fa = args.output_dir / "dti_fa.nii.gz"
    dti_md = args.output_dir / "dti_md.nii.gz"
    run_tensor2metric(dti_d, dti_fa, dti_md)
    print(f"DTI metrics: {dti_fa}, {dti_md}, {dti_s0}")

    # Fit DKI (with kurtosis)
    print("Fitting diffusion kurtosis tensor (DKI)...")
    dki_d = args.output_dir / "dki_d.nii.gz"
    dki_k = args.output_dir / "dki_k.nii.gz"
    dki_s0 = args.output_dir / "dki_s0.nii.gz"
    run_dwi2tensor(dwi_corrected, dki_d, dki_k, dki_s0)
    print(f"DKI tensors: {dki_d}, {dki_k}")

    # Compute DKI metrics
    print("Computing DKI metrics...")
    dki_fa = args.output_dir / "dki_fa.nii.gz"
    dki_md = args.output_dir / "dki_md.nii.gz"
    run_tensor2metric(dki_d, dki_fa, dki_md)
    print(f"DKI metrics: {dki_fa}, {dki_md}, {dki_s0}")

    print("Done")


if __name__ == "__main__":
    main()
