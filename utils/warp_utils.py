"""Warp combination utilities for single-interpolation transforms.

Combines eddy per-volume displacement fields with additional transforms
(e.g., DWI-to-T1w registration) so that all corrections are applied in
a single interpolation step, minimizing data degradation.

Adapted from https://github.com/mrphysics-bonn/AxonDiameter
(function: combine_warps)
Copyright (c) 2024, Marten Veldmann <marten.veldmann@dzne.de>
Licensed under the MIT License (see LICENSES/MIT.txt for full text).
"""

import os
import re
from multiprocessing import Pool
from pathlib import Path

from utils.cmd_utils import run_command


def _get_sorted_eddy_displacement_fields(dfields_dir: Path) -> list[Path]:
    """Get sorted list of eddy displacement field files.

    Parameters
    ----------
    dfields_dir : Path
        Directory containing displacement field files named
        ``dwi_post_eddy.eddy_displacement_fields.<N>.nii.gz``

    Returns
    -------
    list[Path]
        Sorted list of displacement field file paths
    """
    pattern = re.compile(
        r"dwi_post_eddy\.eddy_displacement_fields\.(\d+)\.nii\.gz"
    )
    matched = [
        (int(m.group(1)), dfields_dir / f)
        for f in os.listdir(dfields_dir)
        if (m := pattern.match(f))
    ]
    return [path for _, path in sorted(matched)]


def _process_volume(args: tuple) -> None:
    """Process a single volume: combine warps, apply, and modulate by Jacobian.

    Parameters
    ----------
    args : tuple
        (vol_file, dfield_file, comb_warp_file, jac_file, warped_file, postmat)
    """
    vol_file, dfield_file, comb_warp_file, jac_file, warped_file, postmat = args

    # Combine warps: eddy displacement field + optional affine postmat
    # Reference is always the input volume (matching eddy dfield grid)
    cmd = (
        f"convertwarp"
        f" -o {comb_warp_file}"
        f" -r {vol_file}"
        f" -j {jac_file}"
        f" --warp1={dfield_file}"
    )
    if postmat is not None:
        cmd += f" --postmat={postmat}"
    run_command(cmd, verbose=False)

    # Apply combined warp (output on input volume grid)
    run_command(
        f"applywarp"
        f" -i {vol_file}"
        f" -r {vol_file}"
        f" -o {warped_file}"
        f" -w {comb_warp_file}"
        f" --interp=spline",
        verbose=False,
    )

    # Compute mean Jacobian determinant (average across 3 components)
    run_command(f"fslmaths {jac_file} -Tmean {jac_file}", verbose=False)

    # Jacobian modulation
    run_command(f"fslmaths {warped_file} -mul {jac_file} {warped_file}", verbose=False)


def combine_and_apply_warps(
    eddy_output_dir: Path,
    out_file: Path,
    postmat: Path | None = None,
    nprocs: int | None = None,
) -> Path:
    """Combine eddy displacement fields with additional transforms and apply.

    Splits the eddy outlier-free data into individual volumes, combines
    each volume's eddy displacement field with an optional affine transform
    into a single warp, and applies it with spline interpolation and Jacobian
    modulation. The corrected volumes are merged into a single 4D output.

    Each volume uses itself as the convertwarp/applywarp reference, matching
    the grid on which the eddy displacement fields are defined.

    Parameters
    ----------
    eddy_output_dir : Path
        Directory containing eddy outputs (from ``run_dwifslpreproc``):
        ``eddy_outlier_free_data.nii.gz`` and ``dfields/`` subdirectory
    out_file : Path
        Output path for the combined-warp-corrected 4D DWI
    postmat : Path, optional
        FLIRT-format affine transform to apply after eddy correction
        (e.g., DWI-to-T1w rigid registration from ``convert_ants_to_flirt``)
    nprocs : int, optional
        Number of parallel processes. Defaults to number of physical CPUs.

    Returns
    -------
    Path
        Path to the output file
    """
    eddy_output_dir = Path(eddy_output_dir)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Paths
    dwi = eddy_output_dir / "eddy_outlier_free_data.nii.gz"
    dfields_dir = eddy_output_dir / "dfields"
    work_dir = eddy_output_dir / "warp_combine_work"

    split_dir = work_dir / "split_data"
    comb_warp_dir = work_dir / "comb_warp"
    warp_dir = work_dir / "warped_data"

    for d in [split_dir, comb_warp_dir, warp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not dwi.exists():
        raise FileNotFoundError(
            f"eddy_outlier_free_data.nii.gz not found in {eddy_output_dir}"
        )

    # Split outlier-free DWI into individual volumes
    run_command(f"fslsplit {dwi} {split_dir}/data -t")
    vol_files = sorted(split_dir.glob("data*.nii.gz"))

    # Get sorted displacement fields
    dfield_files = _get_sorted_eddy_displacement_fields(dfields_dir)

    if len(vol_files) != len(dfield_files):
        raise ValueError(
            f"Mismatch: {len(vol_files)} volumes vs "
            f"{len(dfield_files)} displacement fields"
        )

    # Prepare arguments for parallel processing
    process_args = []
    for i, (vol_file, dfield_file) in enumerate(zip(vol_files, dfield_files)):
        process_args.append((
            str(vol_file),
            str(dfield_file),
            str(comb_warp_dir / f"comb_warp_{i:04d}.nii.gz"),
            str(comb_warp_dir / f"jac_{i:04d}.nii.gz"),
            str(warp_dir / f"warped_{i:04d}.nii.gz"),
            str(postmat) if postmat is not None else None,
        ))

    # Process volumes in parallel
    if nprocs is None:
        nprocs = os.cpu_count() or 1
    print(f"Combining warps for {len(vol_files)} volumes using {nprocs} processes...")

    with Pool(processes=nprocs) as pool:
        pool.map(_process_volume, process_args)

    # Merge corrected volumes
    warped_files = sorted(warp_dir.glob("warped_*.nii.gz"))
    warped_list = " ".join(str(f) for f in warped_files)
    run_command(f"fslmerge -t {out_file} {warped_list}")

    print(f"Combined warp output: {out_file}")
    return out_file


def _read_itk_affine(mat_file: Path) -> "np.ndarray":
    """Read an ITK/ANTs ``.mat`` affine transform and return a 4x4 matrix.

    ANTs saves transforms as MATLAB v5 ``.mat`` files containing:
    - ``AffineTransform_double_3_3``: 12 params (9 rotation + 3 translation)
    - ``fixed``: 3 params (center of rotation)

    The full transform is: p' = R @ (p - center) + center + t.
    """
    import numpy as np
    import scipy.io as sio

    data = sio.loadmat(str(mat_file))

    # Find the transform key (may vary: AffineTransform_double_3_3 or _float_3_3)
    transform_key = None
    for key in data:
        if key.startswith("AffineTransform"):
            transform_key = key
            break
    if transform_key is None:
        raise ValueError(f"No AffineTransform found in {mat_file}")

    params = data[transform_key].flatten()
    fixed = data["fixed"].flatten()

    R = params[:9].reshape(3, 3)
    t = params[9:12]
    center = fixed[:3]

    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = t + center - R @ center
    return mat


def _fsl_vox2mm(img: "nib.Nifti1Image") -> "np.ndarray":
    """Compute FSL's internal voxel-to-mm scaling matrix for an image.

    FSL uses a scaled voxel coordinate system. If the sform determinant
    is positive (neurological storage order), FSL flips the x-axis to
    work in its preferred radiological convention.
    """
    import numpy as np

    zooms = np.array(img.header.get_zooms()[:3])
    mat = np.diag([zooms[0], zooms[1], zooms[2], 1.0])

    if np.linalg.det(img.affine[:3, :3]) > 0:
        mat[0, 0] = -zooms[0]
        mat[0, 3] = (img.shape[0] - 1) * zooms[0]

    return mat


def convert_ants_to_flirt(
    ants_mat: Path,
    ref: Path,
    src: Path,
    output: Path,
) -> Path:
    """Convert an ANTs affine transform to FLIRT format.

    Pure-Python replacement for ``c3d_affine_tool -ras2fsl``.  Converts
    an ANTs/ITK ``.mat`` transform to an FSL FLIRT ``.txt`` matrix.

    The conversion chain is::

        FLIRT = FSL_ref @ inv(sform_ref) @ inv(A_ras) @ sform_src @ inv(FSL_src)

    where ``A_ras`` is the ANTs transform converted from LPS to RAS
    (maps fixed/ref → moving/src), so its inverse maps src → ref.

    Parameters
    ----------
    ants_mat : Path
        ANTs affine transform file (e.g., ``*_0GenericAffine.mat``)
    ref : Path
        Reference/target image defining the output space geometry
        (e.g., T1w when registering DWI→T1w)
    src : Path
        Source image defining the input space geometry. When used with
        ``convertwarp --postmat``, this should be in the eddy-undistorted
        DWI space (e.g., first volume of ``dwi_post_eddy.nii.gz``)
    output : Path
        Output FLIRT-format ``.txt`` matrix file

    Returns
    -------
    Path
        Path to the output FLIRT matrix
    """
    import nibabel as nib
    import numpy as np

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Read ITK transform (maps fixed→moving in LPS coordinates)
    itk_mat = _read_itk_affine(ants_mat)

    # Convert LPS→RAS (negate x and y axes)
    c = np.diag([-1.0, -1.0, 1.0, 1.0])
    ras_mat = c @ itk_mat @ c  # maps ref_ras → src_ras

    # Load image headers
    ref_img = nib.load(str(ref))
    src_img = nib.load(str(src))

    # FSL scaling matrices
    fsl_ref = _fsl_vox2mm(ref_img)
    fsl_src = _fsl_vox2mm(src_img)

    # FLIRT = FSL_ref @ inv(sform_ref) @ inv(A_ras) @ sform_src @ inv(FSL_src)
    flirt_mat = (
        fsl_ref
        @ np.linalg.inv(ref_img.affine)
        @ np.linalg.inv(ras_mat)
        @ src_img.affine
        @ np.linalg.inv(fsl_src)
    )

    np.savetxt(output, flirt_mat, fmt="%.10f")
    return output
