"""Common DWI preprocessing functions.

Reusable building blocks for denoising, sidecar copying, and file utilities
used by dataset-specific scripts.
"""

import shutil
from pathlib import Path

from utils.cmd_utils import run_command
from utils.nifti_utils import get_bval_path, get_bvec_path, get_json_path, get_pe_direction, get_readout_time, strip_nifti_ext
from utils.warp_utils import combine_and_apply_warps, convert_ants_to_flirt


def add_suffix(nii_path: Path, suffix: str) -> Path:
    """Add a suffix before .nii.gz, e.g. _denoised.

    Parameters
    ----------
    nii_path : Path
        Path to NIfTI file
    suffix : str
        Suffix to add (e.g. "_denoised", "_noisemap")

    Returns
    -------
    Path
        Path with suffix inserted before .nii.gz
    """
    base = strip_nifti_ext(nii_path)
    return base.with_name(base.name + suffix + ".nii.gz")


def find_dwi_files(input_dir: Path, pattern: str = "*.nii.gz") -> list[Path]:
    """Find NIfTI files in a directory.

    Parameters
    ----------
    input_dir : Path
        Directory to search
    pattern : str
        Glob pattern (default: "*.nii.gz")

    Returns
    -------
    list[Path]
        Sorted list of matching NIfTI paths
    """
    return sorted(input_dir.glob(pattern))


def copy_sidecar(input_nii: Path, output_nii: Path) -> None:
    """Copy bval, bvec, and json sidecars to match the output NIfTI."""
    for get_path in (get_bval_path, get_bvec_path, get_json_path):
        src = get_path(input_nii)
        dst = get_path(output_nii)
        if src.exists():
            shutil.copy2(src, dst)


def denoise_dwi(
    dwi: Path,
    output: Path,
    noisemap: Path | None = None,
    extent: tuple[int, int, int] | None = None,
    nthreads: int | None = None,
) -> None:
    """Run MRtrix dwidenoise (Marchenko-Pastur PCA).

    Parameters
    ----------
    dwi : Path
        Input DWI data
    output : Path
        Output denoised DWI
    noisemap : Path, optional
        Output noise map
    extent : tuple[int, int, int], optional
        Sliding window extent (default: auto, typically 5x5x5)
    nthreads : int, optional
        Number of threads (0=all available)
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = f"dwidenoise {dwi} {output} -force"

    if noisemap is not None:
        cmd += f" -noise {noisemap}"

    if extent is not None:
        cmd += f" -extent {extent[0]},{extent[1]},{extent[2]}"

    if nthreads is not None:
        cmd += f" -nthreads {nthreads}"

    run_command(cmd)


def run_dwifslpreproc(
    dwi: Path,
    dwi_rpe: Path,
    output_dir: Path,
    mode: str = "b0_rpe",
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
        Output directory
    mode : str
        Processing mode (default: "b0_rpe"):
        - "b0_rpe": Use b0 volumes only for fieldmap estimation (-rpe_pair)
        - "full_rpe": Merge all volumes from both PE directions (-rpe_all)
    readout_time : float, optional
        Total readout time in seconds (read from JSON if not provided)
    nthreads : int
        Number of threads for topup (0=all available)
    keep_tmp : bool
        Keep temporary directory for debugging

    Returns
    -------
    tuple[Path, Path]
        (preprocessed DWI path, eddy output directory)
    """
    import glob
    import os
    import tempfile

    if mode not in ["b0_rpe", "full_rpe"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'b0_rpe' or 'full_rpe'")

    # Detect available CPUs if nthreads=0
    if nthreads == 0:
        nthreads = os.cpu_count() or 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename based on mode
    if mode == "b0_rpe":
        # Keep original PE direction in name: *_dir-AP_preprocessed.nii.gz
        output_name = dwi.name.replace("_denoised", "_preprocessed")
    elif mode == "full_rpe":
        # Remove PE direction, indicate merged: *_merged_preprocessed.nii.gz
        if "_dir-" in dwi.name:
            # single_pe_rpe: sub-V06460_acq-b2000n132_dir-AP_denoised.nii.gz
            base_name = dwi.name.split("_dir-")[0]  # Everything before _dir-XX
        else:
            # multi_pe_rpe: dwi_lr_denoised.nii.gz
            base_name = dwi.name.replace("_denoised.nii.gz", "").replace("_denoised.nii", "")
            # Extract just the base without PE direction (dwi_lr -> dwi)
            if base_name.endswith(("_lr", "_rl", "_ap", "_pa")):
                base_name = "_".join(base_name.split("_")[:-1])
        output_name = f"{base_name}_merged_preprocessed.nii.gz"

    output = output_dir / output_name

    # Get bval/bvec files
    bvec = get_bvec_path(dwi)
    bval = get_bval_path(dwi)
    bvec_rpe = get_bvec_path(dwi_rpe)
    bval_rpe = get_bval_path(dwi_rpe)

    # Get PE direction from JSON
    pe_dir = get_pe_direction(dwi)
    if pe_dir is None:
        raise ValueError(f"PE direction not found in JSON sidecar for {dwi}")

    # Get readout time from JSON if not provided
    if readout_time is None:
        readout_time = get_readout_time(dwi)

    if readout_time is None:
        raise ValueError(f"Readout time not found in JSON sidecar for {dwi}")

    # Create temp directory (unique per input file and mode to avoid conflicts)
    input_stem = dwi.stem.replace('.nii', '')
    if keep_tmp:
        tmp_dir = output.parent / f"tmp_{input_stem}_{mode}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"{input_stem}_{mode}_"))

    # Eddy output directory (persisted after processing)
    eddy_output_dir = output_dir / f"{output_name.replace('.nii.gz', '')}_eddy_output"
    # Remove existing eddy output directory to avoid file conflicts
    if eddy_output_dir.exists():
        shutil.rmtree(eddy_output_dir)
    eddy_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Output bvec/bval paths
        out_bvec = get_bvec_path(output)
        out_bval = get_bval_path(output)

        # Scratch path for dwifslpreproc
        scratch_dir = tmp_dir / "dwifslpreproc"

        if mode == "b0_rpe":
            # Extract first b0 from both PE directions
            b0_pe = tmp_dir / "b0_pe.nii.gz"
            b0_rpe = tmp_dir / "b0_rpe.nii.gz"
            run_command(f"mrconvert {dwi} {b0_pe} -force -coord 3 0 -fslgrad {bvec} {bval}", verbose=False)
            run_command(f"mrconvert {dwi_rpe} {b0_rpe} -force -coord 3 0 -fslgrad {bvec_rpe} {bval_rpe}", verbose=False)

            # Merge b0s for topup (PE + RPE)
            b0_pair = tmp_dir / "b0_pair.nii.gz"
            run_command(f"mrcat {b0_pe} {b0_rpe} {b0_pair} -force -axis 3", verbose=False)

            # Run dwifslpreproc with -rpe_pair
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

        elif mode == "full_rpe":
            # Merge full DWI volumes from both PE directions
            merged_dwi = tmp_dir / "merged_dwi.nii.gz"
            run_command(f"mrcat {dwi} {dwi_rpe} {merged_dwi} -force -axis 3", verbose=False)

            # Merge bval files (concatenate)
            merged_bval = tmp_dir / "merged.bval"
            with open(bval, 'r') as f1, open(bval_rpe, 'r') as f2, open(merged_bval, 'w') as out:
                bval1 = f1.read().strip()
                bval2 = f2.read().strip()
                out.write(f"{bval1} {bval2}\n")

            # Merge bvec files (concatenate column-wise)
            import numpy as np
            merged_bvec = tmp_dir / "merged.bvec"
            bvec1 = np.loadtxt(bvec)
            bvec2 = np.loadtxt(bvec_rpe)
            # Handle 1D case (single volume)
            if bvec1.ndim == 1:
                bvec1 = bvec1.reshape(3, 1)
            if bvec2.ndim == 1:
                bvec2 = bvec2.reshape(3, 1)
            merged_bvec_data = np.hstack([bvec1, bvec2])
            np.savetxt(merged_bvec, merged_bvec_data, fmt='%.6f')

            # Run dwifslpreproc with -rpe_all
            cmd = (
                f"dwifslpreproc {merged_dwi} {output} -force "
                f"-nocleanup "
                f"-fslgrad {merged_bvec} {merged_bval} "
                f"-export_grad_fsl {out_bvec} {out_bval} "
                f"-rpe_all -pe_dir {pe_dir} "
                f"-readout_time {readout_time} "
                f"-scratch {scratch_dir} "
                f"-eddyqc_all {eddy_output_dir}"
            )

        cmd += f" -topup_options ' --nthr={nthreads}'"
        cmd += f" -eddy_options ' --repol --data_is_shelled --slm=linear --dfields'"

        run_command(cmd, verbose=True)

        # Copy dfields + dwi_post_eddy from scratch
        dfields_dir = eddy_output_dir / "dfields"
        dfields_dir.mkdir(parents=True, exist_ok=True)

        dfield_files = sorted(glob.glob(
            str(scratch_dir / "dwi_post_eddy.eddy_displacement_fields.*.nii.gz")
        ))
        for f in dfield_files:
            shutil.copy(f, dfields_dir)

        # Copy fully corrected dwi_post_eddy
        dwi_post_eddy = scratch_dir / "dwi_post_eddy.nii.gz"
        if dwi_post_eddy.exists():
            shutil.copy(dwi_post_eddy, eddy_output_dir / "dwi_post_eddy.nii.gz")

    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return output, eddy_output_dir


def create_brain_mask(dwi: Path, output_dir: Path) -> Path:
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

    # Extract mean b=0
    mean_b0 = output_dir / "mean_b0.nii.gz"
    run_command(
        f"dwiextract -bzero -force -fslgrad {bvec} {bval} {dwi} - "
        f"| mrmath - mean -axis 3 {mean_b0} -force",
        verbose=False,
    )

    # Brain extraction with BET
    bet_output = output_dir / "mean_b0_brain"
    run_command(f"bet {mean_b0} {bet_output} -m -R", verbose=False)

    # Rename mask
    brain_mask = output_dir / "brain_mask.nii.gz"
    Path(f"{bet_output}_mask.nii.gz").rename(brain_mask)
    Path(f"{bet_output}.nii.gz").unlink(missing_ok=True)

    # Erode by 1 voxel to tighten the mask
    run_command(f"maskfilter {brain_mask} erode {brain_mask} -force", verbose=False)

    return brain_mask


def _diag_validity_mask(image: Path, mask: Path, diag_low: float, diag_high: float, tmp_dir: Path) -> Path:
    """Create a validity mask from diagonal tensor elements.

    Returns a 3D binary mask where all diagonal elements (volumes 0-2)
    are within [diag_low, diag_high] and the voxel is inside the brain mask.
    Also replaces NaN/Inf with 0 in the image (in-place).
    """
    stem = image.stem.replace(".nii", "")
    # Replace NaN/Inf with 0
    run_command(f"mrcalc {image} -finite {image} 0 -if {image} -force", verbose=False)
    # Extract diagonal volumes (0, 1, 2)
    diag = tmp_dir / f"_diag_{stem}.nii.gz"
    run_command(f"mrconvert {image} -coord 3 0:2 {diag} -force", verbose=False)
    # Per-voxel validity: all diags in [low, high]
    valid = tmp_dir / f"_valid_{stem}.nii.gz"
    run_command(
        f"mrcalc {diag} {diag_low} -ge {diag} {diag_high} -le -mult "
        f"{mask} -mult {valid} -force",
        verbose=False,
    )
    # Collapse across volumes: valid only if ALL diags pass
    valid_mask = tmp_dir / f"_valid_mask_{stem}.nii.gz"
    run_command(f"mrmath {valid} min -axis 3 {valid_mask} -force", verbose=False)
    # Clean up intermediates
    diag.unlink(missing_ok=True)
    valid.unlink(missing_ok=True)
    return valid_mask


def _kt_contract_nnnn(kt, n):
    """Contract kurtosis tensor with a single direction: Σ W_ijkl n_i n_j n_k n_l.

    Parameters
    ----------
    kt : ndarray, shape (N, 15)
        Kurtosis tensor elements (MRtrix ordering)
    n : ndarray, shape (N, 3)
        Direction vectors

    Returns
    -------
    ndarray, shape (N,)
    """
    n1, n2, n3 = n[:, 0], n[:, 1], n[:, 2]
    # MRtrix KT ordering with combinatorial multiplicities:
    #  0-2: W_aaaa (×1), 3-8: W_aaab (×4), 9-11: W_aabb (×6), 12-14: W_aabc (×12)
    return (
        kt[:, 0]*n1**4 + kt[:, 1]*n2**4 + kt[:, 2]*n3**4
        + 4*(kt[:, 3]*n1**3*n2 + kt[:, 4]*n1**3*n3
             + kt[:, 5]*n1*n2**3 + kt[:, 6]*n2**3*n3
             + kt[:, 7]*n1*n3**3 + kt[:, 8]*n2*n3**3)
        + 6*(kt[:, 9]*n1**2*n2**2 + kt[:, 10]*n1**2*n3**2
             + kt[:, 11]*n2**2*n3**2)
        + 12*(kt[:, 12]*n1**2*n2*n3
              + kt[:, 13]*n1*n2*n3**2
              + kt[:, 14]*n1*n2**2*n3)
    )


def _kt_contract_uuvv(kt, u, v):
    """Contract kurtosis tensor with two direction pairs: Σ W_ijkl u_i u_j v_k v_l.

    Parameters
    ----------
    kt : ndarray, shape (N, 15)
        Kurtosis tensor elements (MRtrix ordering)
    u, v : ndarray, shape (N, 3)
        Direction vectors

    Returns
    -------
    ndarray, shape (N,)
    """
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    v1, v2, v3 = v[:, 0], v[:, 1], v[:, 2]
    return (
        # W_aaaa: u_a² v_a²
        kt[:, 0]*u1**2*v1**2 + kt[:, 1]*u2**2*v2**2 + kt[:, 2]*u3**2*v3**2
        # W_aaab (×4 perms → 2 u_a² v_a v_b + 2 u_a u_b v_a²)
        + kt[:, 3]*(2*u1**2*v1*v2 + 2*u1*u2*v1**2)      # W1112: a=1,b=2
        + kt[:, 4]*(2*u1**2*v1*v3 + 2*u1*u3*v1**2)      # W1113: a=1,b=3
        + kt[:, 5]*(2*u2**2*v1*v2 + 2*u1*u2*v2**2)      # W1222: a=2,b=1
        + kt[:, 6]*(2*u2**2*v2*v3 + 2*u2*u3*v2**2)      # W2223: a=2,b=3
        + kt[:, 7]*(2*u3**2*v1*v3 + 2*u1*u3*v3**2)      # W1333: a=3,b=1
        + kt[:, 8]*(2*u3**2*v2*v3 + 2*u2*u3*v3**2)      # W2333: a=3,b=2
        # W_aabb (×6 perms → u_a² v_b² + 4 u_a u_b v_a v_b + u_b² v_a²)
        + kt[:, 9]*(u1**2*v2**2 + 4*u1*u2*v1*v2 + u2**2*v1**2)    # W1122
        + kt[:, 10]*(u1**2*v3**2 + 4*u1*u3*v1*v3 + u3**2*v1**2)   # W1133
        + kt[:, 11]*(u2**2*v3**2 + 4*u2*u3*v2*v3 + u3**2*v2**2)   # W2233
        # W_aabc (×12 perms → 2 u_a² v_b v_c + 4 u_a u_b v_a v_c
        #                    + 4 u_a u_c v_a v_b + 2 u_b u_c v_a²)
        + kt[:, 12]*(2*u1**2*v2*v3 + 4*u1*u2*v1*v3      # W1123: a=1,b=2,c=3
                     + 4*u1*u3*v1*v2 + 2*u2*u3*v1**2)
        + kt[:, 13]*(2*u3**2*v1*v2 + 4*u1*u3*v2*v3      # W1233: a=3,b=1,c=2
                     + 4*u2*u3*v1*v3 + 2*u1*u2*v3**2)
        + kt[:, 14]*(2*u2**2*v1*v3 + 4*u1*u2*v2*v3      # W1223: a=2,b=1,c=3
                     + 4*u2*u3*v1*v2 + 2*u1*u3*v2**2)
    )


def _tabesh_F1F2(la, lb, lc):
    """Compute F1 and F2 coefficients from Tabesh et al. (2011).

    For eigenvalue permutation (la, lb, lc):
    - F1 is the weight for Ŵ_aaaa (diagonal kurtosis along eigenvector a)
    - F2 is the weight for Ŵ_bbcc (cross-kurtosis between eigenvectors b,c)

    Parameters
    ----------
    la, lb, lc : ndarray, shape (N,)
        Diffusion tensor eigenvalues (one specific permutation)

    Returns
    -------
    F1, F2 : ndarray, shape (N,)
    """
    import numpy as np
    from scipy.special import elliprf, elliprd

    l_sum_sq = (la + lb + lc) ** 2
    sqrt_lblc = np.sqrt(lb * lc)

    rf = elliprf(la / lb, la / lc, np.ones_like(la))
    rd = elliprd(la / lb, la / lc, np.ones_like(la))

    # F1 (Tabesh Eq. A1)
    F1 = (l_sum_sq / (18 * (la - lb) * (la - lc))) * (
        sqrt_lblc / la * rf
        + (3*la**2 - la*lb - la*lc - lb*lc) / (3*la*sqrt_lblc) * rd
        - 1
    )

    # F2 (Tabesh Eq. A2)
    F2 = (l_sum_sq / (3 * (lb - lc)**2)) * (
        (lb + lc) / sqrt_lblc * rf
        + (2*la - lb - lc) / (3*sqrt_lblc) * rd
        - 2
    )

    return F1, F2


def _compute_mean_kurtosis(
    dt_file: Path, kt_file: Path, mask_file: Path, output: Path,
) -> None:
    """Compute mean kurtosis (MK) using the Tabesh et al. (2011) analytical formula.

    Closed-form expression using Carlson's elliptic integrals RF and RD.
    The diffusion tensor is eigendecomposed, the kurtosis tensor is rotated
    into the eigenvector frame, and MK is computed as a weighted sum of
    6 rotated kurtosis components (3 diagonal Ŵ_aaaa + 3 cross Ŵ_aabb).

    Reference: Tabesh et al., "Estimation of tensors and tensor-derived
    measures in diffusional kurtosis imaging", MRM 65:823-836, 2011.

    Parameters
    ----------
    dt_file : Path
        Diffusion tensor (6 volumes, MRtrix order: D11, D22, D33, D12, D13, D23)
    kt_file : Path
        Kurtosis tensor (15 volumes, MRtrix order)
    mask_file : Path
        Brain mask
    output : Path
        Output MK image path
    """
    import nibabel as nib
    import numpy as np

    mask_img = nib.load(str(mask_file))
    mask_data = mask_img.get_fdata() > 0
    dt_img = nib.load(str(dt_file))
    kt_img = nib.load(str(kt_file))
    dt_data = dt_img.get_fdata()
    kt_data = kt_img.get_fdata()

    # Dimension check
    if mask_data.shape != dt_data.shape[:3]:
        raise ValueError(
            f"Mask shape {mask_data.shape} doesn't match tensor spatial dims {dt_data.shape[:3]}\n"
            f"  Mask file: {mask_file}\n"
            f"  DT file: {dt_file}\n"
            f"  Mask affine:\n{mask_img.affine}\n"
            f"  DT affine:\n{dt_img.affine}"
        )

    # Extract masked voxels
    idx = np.where(mask_data)
    N = len(idx[0])
    dt_m = dt_data[idx]  # (N, 6)
    kt_m = kt_data[idx]  # (N, 15)

    # Build symmetric 3×3 diffusion tensor and eigendecompose
    D = np.zeros((N, 3, 3))
    D[:, 0, 0] = dt_m[:, 0]
    D[:, 1, 1] = dt_m[:, 1]
    D[:, 2, 2] = dt_m[:, 2]
    D[:, 0, 1] = D[:, 1, 0] = dt_m[:, 3]
    D[:, 0, 2] = D[:, 2, 0] = dt_m[:, 4]
    D[:, 1, 2] = D[:, 2, 1] = dt_m[:, 5]

    evals, evecs = np.linalg.eigh(D)  # ascending order
    evals = np.maximum(evals, 1e-10)  # clamp to positive

    # Eigenvectors: evecs[:, :, k] is the k-th eigenvector
    e1, e2, e3 = evecs[:, :, 0], evecs[:, :, 1], evecs[:, :, 2]

    # Rotated kurtosis tensor components in eigenvector frame
    W1111 = _kt_contract_nnnn(kt_m, e1)
    W2222 = _kt_contract_nnnn(kt_m, e2)
    W3333 = _kt_contract_nnnn(kt_m, e3)
    W1122 = _kt_contract_uuvv(kt_m, e1, e2)
    W1133 = _kt_contract_uuvv(kt_m, e1, e3)
    W2233 = _kt_contract_uuvv(kt_m, e2, e3)

    # Add relative jitter to eigenvalues to avoid 0/0 in F1/F2 denominators
    # (degenerate when any two eigenvalues are equal)
    l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]
    eps = 1e-7 * np.maximum((l1 + l2 + l3) / 3, 1e-15)
    l1, l2, l3 = l1 + 3*eps, l2 + 2*eps, l3 + eps

    # F1, F2 for three permutations (Tabesh Eq. 16)
    f1_123, f2_123 = _tabesh_F1F2(l1, l2, l3)
    f1_213, f2_213 = _tabesh_F1F2(l2, l1, l3)
    f1_321, f2_321 = _tabesh_F1F2(l3, l2, l1)

    # MK = Σ F1 × Ŵ_diag + Σ F2 × Ŵ_cross
    mk_m = (f1_123 * W1111 + f2_123 * W2233
            + f1_213 * W2222 + f2_213 * W1133
            + f1_321 * W3333 + f2_321 * W1122)

    # Place back into volume
    mk = np.zeros(dt_data.shape[:3], dtype=np.float64)
    mk[idx] = mk_m

    mk_img = nib.Nifti1Image(mk.astype(np.float32), mask_img.affine)
    nib.save(mk_img, str(output))


def _threshold_metric(image: Path, low: float, high: float | None = None) -> None:
    """Zero out voxels outside [low, high] range (in-place).

    If high is None, only apply lower bound.
    """
    if high is not None:
        run_command(
            f"mrcalc {image} {low} -ge {image} {high} -le -mult {image} -mult {image} -force",
            verbose=False,
        )
    else:
        run_command(
            f"mrcalc {image} {low} -ge {image} -mult {image} -force",
            verbose=False,
        )


def fit_tensors(dwi: Path, output_prefix: Path, mask: Path) -> dict[str, Path]:
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
    output_prefix : Path
        Output file prefix (e.g., /path/to/sub-V06460_dir-AP_preprocessed)
    mask : Path
        Brain mask image

    Returns
    -------
    dict[str, Path]
        Dictionary with paths to all output tensors and metrics
    """
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    bvec = get_bvec_path(dwi)
    bval = get_bval_path(dwi)

    outputs = {}

    # DTI - extract only b < 1700 for fitting (DTI assumes Gaussian diffusion)
    print("    Fitting DTI (b < 1700)...")
    import numpy as np

    # Determine which shells to keep (b < 1700)
    bvals = np.loadtxt(bval)
    unique_bvals = np.unique(np.round(bvals / 100) * 100)  # Round to nearest 100
    low_b_shells = unique_bvals[unique_bvals < 1700]
    shells_str = ",".join(str(int(b)) for b in low_b_shells)

    # Extract low-b shells for DTI
    dwi_dti = tmp_dir / "dwi_for_dti.nii.gz"
    bval_dti = tmp_dir / "dwi_for_dti.bval"
    bvec_dti = tmp_dir / "dwi_for_dti.bvec"

    run_command(
        f"dwiextract {dwi} {dwi_dti}"
        f" -shells {shells_str}"
        f" -fslgrad {bvec} {bval}"
        f" -export_grad_fsl {bvec_dti} {bval_dti}"
        f" -force",
        verbose=False,
    )

    n_vols_dti = len(np.loadtxt(bval_dti))
    print(f"      Using {n_vols_dti}/{len(bvals)} volumes (shells: {shells_str})")

    dti_d = Path(f"{output_prefix}_dti_d.nii.gz")
    dti_s0 = Path(f"{output_prefix}_dti_s0.nii.gz")

    cmd = f"dwi2tensor {dwi_dti} {dti_d} -force -fslgrad {bvec_dti} {bval_dti} -iter 10 -mask {mask} -b0 {dti_s0}"
    run_command(cmd, verbose=False)

    # Clean up DTI temp files
    dwi_dti.unlink(missing_ok=True)
    bval_dti.unlink(missing_ok=True)
    bvec_dti.unlink(missing_ok=True)

    print("    Cleaning diffusion tensor...")
    dt_valid = _diag_validity_mask(dti_d, mask, diag_low=0, diag_high=0.003, tmp_dir=tmp_dir)
    run_command(f"mrcalc {dti_d} {dt_valid} -mult {dti_d} -force", verbose=False)
    run_command(f"mrcalc {dti_s0} {dt_valid} -mult {dti_s0} -force", verbose=False)
    dt_valid.unlink(missing_ok=True)

    dti_fa = Path(f"{output_prefix}_dti_fa.nii.gz")
    dti_md = Path(f"{output_prefix}_dti_md.nii.gz")
    cmd = f"tensor2metric {dti_d} -force -fa {dti_fa} -adc {dti_md}"
    run_command(cmd, verbose=False)

    print("    Filtering DTI metrics...")
    _threshold_metric(dti_fa, 0, 1)
    _threshold_metric(dti_md, 0, 0.003)
    result = run_command(
        f"mrstats {dti_s0} -mask {mask} -output mean", verbose=False,
    )
    s0_mean = float(result.stdout.strip())
    _threshold_metric(dti_s0, 0, s0_mean * 3)

    outputs.update({
        "dti_d": dti_d,
        "dti_fa": dti_fa,
        "dti_md": dti_md,
        "dti_s0": dti_s0,
    })

    # DKI
    print("    Fitting DKI...")
    dki_d = Path(f"{output_prefix}_dki_d.nii.gz")
    dki_k = Path(f"{output_prefix}_dki_k.nii.gz")
    dki_s0 = Path(f"{output_prefix}_dki_s0.nii.gz")

    cmd = f"dwi2tensor {dwi} {dki_d} -force -fslgrad {bvec} {bval} -iter 10 -mask {mask} -dkt {dki_k} -b0 {dki_s0}"
    run_command(cmd, verbose=False)

    print("    Cleaning tensors...")
    d_valid = _diag_validity_mask(dki_d, mask, diag_low=0, diag_high=0.003, tmp_dir=tmp_dir)
    k_valid = _diag_validity_mask(dki_k, mask, diag_low=0, diag_high=10, tmp_dir=tmp_dir)
    # Combined mask: voxel must pass both D and K checks
    combined_valid = tmp_dir / "_combined_valid.nii.gz"
    run_command(f"mrcalc {d_valid} {k_valid} -mult {combined_valid} -force", verbose=False)
    d_valid.unlink(missing_ok=True)
    k_valid.unlink(missing_ok=True)
    # Apply combined mask to D, K, and S0
    run_command(f"mrcalc {dki_d} {combined_valid} -mult {dki_d} -force", verbose=False)
    run_command(f"mrcalc {dki_k} {combined_valid} -mult {dki_k} -force", verbose=False)
    run_command(f"mrcalc {dki_s0} {combined_valid} -mult {dki_s0} -force", verbose=False)
    combined_valid.unlink(missing_ok=True)

    dki_fa = Path(f"{output_prefix}_dki_fa.nii.gz")
    dki_md = Path(f"{output_prefix}_dki_md.nii.gz")
    cmd = f"tensor2metric {dki_d} -force -fa {dki_fa} -adc {dki_md}"
    run_command(cmd, verbose=False)

    # Compute mean kurtosis via directional averaging over the unit sphere
    print("    Computing mean kurtosis...")
    dki_mk = Path(f"{output_prefix}_dki_mk.nii.gz")
    _compute_mean_kurtosis(dki_d, dki_k, mask, dki_mk)

    print("    Filtering DKI metrics...")
    _threshold_metric(dki_fa, 0, 1)
    _threshold_metric(dki_md, 0, 0.003)
    result = run_command(
        f"mrstats {dki_s0} -mask {mask} -output mean", verbose=False,
    )
    s0_mean = float(result.stdout.strip())
    _threshold_metric(dki_s0, 0, s0_mean * 3)
    _threshold_metric(dki_mk, 0, 3)

    outputs.update({
        "dki_d": dki_d,
        "dki_k": dki_k,
        "dki_fa": dki_fa,
        "dki_md": dki_md,
        "dki_mk": dki_mk,
        "dki_s0": dki_s0,
    })

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return outputs


def register_to_anat_ref(
    preprocessed_dwi: Path,
    eddy_output_dir: Path,
    anat_ref_image: Path,
    output_dir: Path,
    brain_mask: Path | None = None,
    anat_mask_image: Path | None = None,
    dwi_mask: Path | None = None,
    nthreads: int = 0,
) -> tuple[Path, Path]:
    """Register DWI to anatomical reference with single-interpolation warp combination.

    Performs rigid registration of DWI → anatomical reference, then combines
    eddy displacement fields + registration transform in a single interpolation step.

    Parameters
    ----------
    preprocessed_dwi : Path
        Preprocessed DWI from dwifslpreproc (eddy-corrected)
    eddy_output_dir : Path
        Directory containing eddy outputs (dwi_post_eddy.nii.gz and dfields/)
    anat_ref_image : Path
        Anatomical reference image (e.g., T1w, MTsat)
    output_dir : Path
        Output directory for registered DWI and intermediate files
    brain_mask : Path, optional
        Brain mask for tensor fitting (if None, will create one)
    anat_mask_image : Path, optional
        Anatomical brain mask to regrid and use (instead of creating from DWI)
    dwi_mask : Path, optional
        DWI brain mask to use for masked registration
    nthreads : int
        Number of threads (0=all available)

    Returns
    -------
    tuple[Path, Path]
        (registered DWI path, brain mask path)
    """
    import os

    if nthreads == 0:
        nthreads = os.cpu_count() or 1
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_dir = output_dir / "reg"
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Extract first volume of dwi_post_eddy with fslroi to preserve
    # FSL headers (consistent with eddy displacement fields)
    print("  Extracting first volume from dwi_post_eddy...")
    eddy_post_vol0 = reg_dir / "dwi_post_eddy_vol0.nii.gz"
    dwi_post_eddy = eddy_output_dir / "dwi_post_eddy.nii.gz"
    run_command(f"fslroi {dwi_post_eddy} {eddy_post_vol0} 0 1", verbose=False)

    # Register eddy first vol → anatomical reference (rigid, mutual information)
    # ANTs threading is controlled via environment variable, not CLI flag
    print(f"  Registering DWI → anatomical reference (rigid, {nthreads} threads)...")
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    reg_prefix = reg_dir / "b0_to_anat"

    # Apply masks directly to images before registration if available
    anat_for_reg = anat_ref_image
    dwi_for_reg = eddy_post_vol0

    if anat_mask_image is not None and dwi_mask is not None:
        print(f"    Masking images for registration")

        # Regrid anatomical mask to match anatomical reference
        anat_mask_reg = reg_dir / "anat_mask_for_reg.nii.gz"
        run_command(
            f"mrtransform {anat_mask_image} -template {anat_ref_image} "
            f"-interp nearest {anat_mask_reg} -force",
            verbose=False,
        )
        run_command(
            f"mrconvert {anat_mask_reg} -strides {anat_ref_image} {anat_mask_reg} -force",
            verbose=False,
        )

        # Regrid DWI mask to match eddy_post_vol0
        dwi_mask_reg = reg_dir / "dwi_mask_for_reg.nii.gz"
        run_command(
            f"mrtransform {dwi_mask} -template {eddy_post_vol0} "
            f"-interp nearest {dwi_mask_reg} -force",
            verbose=False,
        )
        run_command(
            f"mrconvert {dwi_mask_reg} -strides {eddy_post_vol0} {dwi_mask_reg} -force",
            verbose=False,
        )

        # Create masked anatomical reference
        anat_masked = reg_dir / "anat_masked.nii.gz"
        run_command(
            f"mrcalc {anat_ref_image} {anat_mask_reg} -mult {anat_masked} -force",
            verbose=False,
        )

        # Create masked DWI
        dwi_masked = reg_dir / "dwi_masked.nii.gz"
        run_command(
            f"mrcalc {eddy_post_vol0} {dwi_mask_reg} -mult {dwi_masked} -force",
            verbose=False,
        )

        anat_for_reg = anat_masked
        dwi_for_reg = dwi_masked

    run_command(
        f"antsRegistration --dimensionality 3 --float 0"
        f" --output [{reg_prefix}_,{reg_prefix}_Warped.nii.gz]"
        f" --interpolation Linear"
        f" --use-histogram-matching 0"
        f" --initial-moving-transform [{anat_for_reg},{dwi_for_reg},1]"
        f" --transform Rigid[0.1]"
        f" --metric MI[{anat_for_reg},{dwi_for_reg},1,32,Regular,0.25]"
        f" --convergence [1000x500x250x100,1e-6,10]"
        f" --smoothing-sigmas 3x2x1x0vox"
        f" --shrink-factors 8x4x2x1",
        verbose=False,
    )
    ants_mat = Path(f"{reg_prefix}_0GenericAffine.mat")
    print(f"    ANTs transform: {ants_mat.name}")

    # Convert ANTs transform → FLIRT format
    # Use eddy_post_vol0 for both ref and src to keep FSL headers
    # consistent with the eddy displacement fields in convertwarp
    print("  Converting ANTs transform to FLIRT format...")
    flirt_mat = reg_dir / "b0_to_anat_flirt.txt"
    convert_ants_to_flirt(
        ants_mat, ref=eddy_post_vol0, src=eddy_post_vol0, output=flirt_mat,
    )

    # Combine eddy warps + registration in single interpolation
    # Output matches anatomical reference voxel size and grid
    # Using MRtrix mrtransform handles stride order correctly
    print("  Applying combined warps (eddy + registration)...")
    dwi_anat = output_dir / f"{preprocessed_dwi.stem.replace('.nii', '')}_anat.nii.gz"
    combine_and_apply_warps(
        eddy_output_dir, dwi_anat,
        postmat=flirt_mat,
        ref_image=anat_ref_image,
        nprocs=nthreads,
    )

    # Copy bvecs/bvals to anatomical-space output
    # combine_and_apply_warps uses eddy_outlier_free_data which has the
    # pre-recombination volume count (e.g., 264 for -rpe_all).
    # The preprocessed DWI may have fewer volumes (132 after recombination).
    # If so, duplicate bvec/bval to match the output volume count.
    import nibabel as nib
    import numpy as np

    bvec_src = get_bvec_path(preprocessed_dwi)
    bval_src = get_bval_path(preprocessed_dwi)
    n_vols = nib.load(str(dwi_anat)).shape[3]
    bvecs = np.loadtxt(bvec_src)
    bvals = np.loadtxt(bval_src)
    n_grad = bvecs.shape[1] if bvecs.ndim == 2 else 1

    if n_vols == n_grad:
        shutil.copy(bvec_src, get_bvec_path(dwi_anat))
        shutil.copy(bval_src, get_bval_path(dwi_anat))
    elif n_vols == 2 * n_grad:
        # -rpe_all: same directions acquired twice, duplicate bvec/bval
        bvecs_dup = np.hstack([bvecs, bvecs])
        bvals_dup = np.concatenate([bvals, bvals])
        np.savetxt(get_bvec_path(dwi_anat), bvecs_dup, fmt="%.6f")
        np.savetxt(get_bval_path(dwi_anat), bvals_dup.reshape(1, -1), fmt="%.0f")
    else:
        raise ValueError(
            f"Volume count mismatch: output has {n_vols} volumes "
            f"but bvec/bval has {n_grad} entries"
        )
    print(f"    Registered DWI: {dwi_anat.name} ({n_vols} volumes)")

    # Brain masking
    if brain_mask is None:
        if anat_mask_image is not None and anat_mask_image.exists():
            # Regrid anatomical mask to match registered DWI
            print("  Regridding anatomical mask to registered DWI space...")
            final_mask = output_dir / "brain_mask.nii.gz"
            mask_tmp = output_dir / "_tmp_mask_anat.nii.gz"
            run_command(
                f"mrtransform {anat_mask_image} -template {dwi_anat} "
                f"-interp nearest {mask_tmp} -force",
                verbose=False,
            )
            run_command(
                f"mrconvert {mask_tmp} -strides {dwi_anat} {final_mask} -force",
                verbose=False,
            )
            mask_tmp.unlink(missing_ok=True)
            brain_mask = final_mask
            print(f"    Brain mask (from anat): {final_mask.name}")
        else:
            # Create brain mask from registered DWI
            print("  Creating brain mask from registered DWI...")
            mask_tmp = output_dir / "_mask_tmp"
            brain_mask = create_brain_mask(dwi_anat, mask_tmp)
            final_mask = output_dir / "brain_mask.nii.gz"
            brain_mask.rename(final_mask)
            shutil.rmtree(mask_tmp, ignore_errors=True)
            brain_mask = final_mask
            print(f"    Brain mask (from DWI): {final_mask.name}")

    # Create masked DWI
    masked_dwi = output_dir / f"{dwi_anat.stem.replace('.nii', '')}_masked.nii.gz"
    run_command(f"mrcalc {dwi_anat} {brain_mask} -mult {masked_dwi} -force", verbose=False)
    print(f"    Masked DWI: {masked_dwi.name}")

    return dwi_anat, brain_mask


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
    keep_tmp: bool = False,
) -> None:
    """Process DWI pair with a specific mode (b0_rpe or full_rpe).

    High-level orchestration function that runs the full DWI processing pipeline:
    preprocessing → brain masking → tensor fitting → registration → segmentation.

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
    keep_tmp : bool
        Keep temporary directory (contains scratch dir with eddy QC outputs)
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
            dwi_forward, dwi_reverse, dwi_dir, mode=mode, nthreads=nthreads, keep_tmp=keep_tmp
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
