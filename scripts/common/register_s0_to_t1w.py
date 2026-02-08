"""Register S0 to T1w using ANTs.

This script performs registration of S0 images (b=0 from DWI) to T1w structural
images using ANTs command-line tools. It can also apply the computed transforms
to other derived images (FA, MD, etc.).
"""

import argparse
from pathlib import Path

from utils.cmd_utils import run_command


def run_ants_registration(
    fixed: Path,
    moving: Path,
    output_prefix: Path,
    transform_type: str = "a",
    metric: str = "MI",
    convergence: str = "[1000x500x250x100,1e-6,10]",
    smoothing_sigmas: str = "3x2x1x0vox",
    shrink_factors: str = "8x4x2x1",
    use_histogram_matching: bool = True,
    interpolation: str = "Linear",
    fixed_mask: Path | None = None,
    moving_mask: Path | None = None,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Run ANTs registration.

    Parameters
    ----------
    fixed : Path
        Fixed/reference image (T1w)
    moving : Path
        Moving image to register (S0)
    output_prefix : Path
        Output prefix for transforms and registered image
    transform_type : str
        Transform type: 'r' (rigid), 'a' (affine), 's' (non-linear SyN)
        Default: 'a' (affine)
    metric : str
        Similarity metric: 'MI' (mutual information), 'CC' (cross-correlation),
        'MeanSquares', 'Demons', 'GC' (global correlation)
        Default: 'MI'
    convergence : str
        Convergence criteria [iterations,tolerance,window]
        Default: '[1000x500x250x100,1e-6,10]'
    smoothing_sigmas : str
        Smoothing sigmas at each level
        Default: '3x2x1x0vox'
    shrink_factors : str
        Shrink factors at each level
        Default: '8x4x2x1'
    use_histogram_matching : bool
        Use histogram matching before registration
        Default: True
    interpolation : str
        Interpolation method for output: 'Linear', 'NearestNeighbor',
        'BSpline', 'Gaussian'
        Default: 'Linear'
    fixed_mask : Path, optional
        Mask for fixed image (limits metric computation region)
    moving_mask : Path, optional
        Mask for moving image (limits metric computation region)
    verbose : bool
        Print verbose output
        Default: False

    Returns
    -------
    tuple[Path, Path]
        (registered image path, transform path)
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Build antsRegistration command
    cmd = f"antsRegistration --dimensionality 3 --float 0"

    # Output naming
    cmd += f" --output [{output_prefix}_,{output_prefix}_Warped.nii.gz]"

    # Interpolation
    cmd += f" --interpolation {interpolation}"

    # Use histogram matching
    if use_histogram_matching:
        cmd += " --use-histogram-matching 1"
    else:
        cmd += " --use-histogram-matching 0"

    # Masks (if provided)
    if fixed_mask is not None:
        cmd += f" --masks [{fixed_mask},{moving_mask if moving_mask else fixed_mask}]"

    # Initial moving transform (center of mass alignment)
    cmd += f" --initial-moving-transform [{fixed},{moving},1]"

    # Registration stages based on transform type
    if transform_type in ["r", "a", "s"]:
        # Rigid stage (always included)
        cmd += (
            f" --transform Rigid[0.1]"
            f" --metric {metric}[{fixed},{moving},1,32,Regular,0.25]"
            f" --convergence {convergence}"
            f" --smoothing-sigmas {smoothing_sigmas}"
            f" --shrink-factors {shrink_factors}"
        )

    if transform_type in ["a", "s"]:
        # Affine stage
        cmd += (
            f" --transform Affine[0.1]"
            f" --metric {metric}[{fixed},{moving},1,32,Regular,0.25]"
            f" --convergence {convergence}"
            f" --smoothing-sigmas {smoothing_sigmas}"
            f" --shrink-factors {shrink_factors}"
        )

    if transform_type == "s":
        # Non-linear SyN stage
        syn_convergence = "[100x70x50x20,1e-6,10]"
        syn_smoothing = "3x2x1x0vox"
        syn_shrink = "8x4x2x1"
        cmd += (
            f" --transform SyN[0.1,3,0]"
            f" --metric CC[{fixed},{moving},1,4]"
            f" --convergence {syn_convergence}"
            f" --smoothing-sigmas {syn_smoothing}"
            f" --shrink-factors {syn_shrink}"
        )

    # Verbosity
    if verbose:
        cmd += " --verbose 1"

    run_command(cmd)

    # Determine transform file(s)
    if transform_type == "r":
        transform = Path(f"{output_prefix}_0GenericAffine.mat")
    elif transform_type == "a":
        transform = Path(f"{output_prefix}_0GenericAffine.mat")
    elif transform_type == "s":
        # For SyN, we have multiple transforms (warp + affine)
        transform = output_prefix.parent / f"{output_prefix.name}_transforms"

    registered = Path(f"{output_prefix}_Warped.nii.gz")

    return registered, transform


def apply_transforms(
    fixed: Path,
    moving: Path,
    output: Path,
    transforms: list[Path],
    interpolation: str = "Linear",
    verbose: bool = False,
) -> Path:
    """Apply ANTs transforms to an image.

    Parameters
    ----------
    fixed : Path
        Reference image (defines output space)
    moving : Path
        Image to transform
    output : Path
        Output transformed image
    transforms : list[Path]
        List of transform files (applied in reverse order)
    interpolation : str
        Interpolation method: 'Linear', 'NearestNeighbor', 'BSpline', 'Gaussian'
        Default: 'Linear'
    verbose : bool
        Print verbose output
        Default: False

    Returns
    -------
    Path
        Output image path
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"antsApplyTransforms -d 3 -e 0"
        f" -i {moving}"
        f" -r {fixed}"
        f" -o {output}"
        f" -n {interpolation}"
    )

    # Add transforms (applied in reverse order)
    for t in transforms:
        cmd += f" -t {t}"

    if verbose:
        cmd += " -v 1"

    run_command(cmd)

    return output


def get_transform_files(output_prefix: Path, transform_type: str) -> list[Path]:
    """Get list of transform files based on registration type.

    Parameters
    ----------
    output_prefix : Path
        Output prefix used in registration
    transform_type : str
        Transform type: 'r', 'a', or 's'

    Returns
    -------
    list[Path]
        List of transform files (in order for antsApplyTransforms)
    """
    if transform_type in ["r", "a"]:
        # Rigid or affine: single .mat file
        return [Path(f"{output_prefix}_0GenericAffine.mat")]
    elif transform_type == "s":
        # SyN: warp field + affine
        # Note: antsApplyTransforms applies them in reverse order
        return [
            Path(f"{output_prefix}_1Warp.nii.gz"),
            Path(f"{output_prefix}_0GenericAffine.mat"),
        ]
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Register S0 to T1w using ANTs (Rigid + MI)"
    )
    parser.add_argument(
        "--s0",
        type=Path,
        required=True,
        help="Input S0 image (moving image)",
    )
    parser.add_argument(
        "--t1w",
        type=Path,
        required=True,
        help="T1w structural image (fixed/reference image)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for registered images and transforms",
    )
    parser.add_argument(
        "--apply-to",
        type=Path,
        nargs="+",
        help="Additional images to transform (e.g., FA, MD maps)",
    )
    parser.add_argument(
        "--fixed-mask",
        type=Path,
        help="Mask for fixed image (T1w brain mask)",
    )
    parser.add_argument(
        "--moving-mask",
        type=Path,
        help="Mask for moving image (S0 brain mask)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output from ANTs",
    )

    args = parser.parse_args()

    if not args.s0.exists():
        raise FileNotFoundError(f"S0 image not found: {args.s0}")
    if not args.t1w.exists():
        raise FileNotFoundError(f"T1w image not found: {args.t1w}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed parameters optimized for S0->T1w registration
    transform_type = "r"  # Rigid
    metric = "MI"  # Mutual information
    interpolation = "Linear"

    print(f"Registering S0 to T1w using ANTs (Rigid + MI)")
    print(f"S0 (moving):   {args.s0}")
    print(f"T1w (fixed):   {args.t1w}")
    print(f"Output dir:    {args.output_dir}")

    # Output prefix for registration
    output_prefix = args.output_dir / "s0_to_t1w"

    # Validate masks if provided
    if args.fixed_mask and not args.fixed_mask.exists():
        raise FileNotFoundError(f"Fixed mask not found: {args.fixed_mask}")
    if args.moving_mask and not args.moving_mask.exists():
        raise FileNotFoundError(f"Moving mask not found: {args.moving_mask}")

    if args.fixed_mask:
        print(f"Fixed mask:    {args.fixed_mask}")
    if args.moving_mask:
        print(f"Moving mask:   {args.moving_mask}")

    # Run registration
    print("\nRunning registration...")
    registered, _ = run_ants_registration(
        fixed=args.t1w,
        moving=args.s0,
        output_prefix=output_prefix,
        transform_type=transform_type,
        metric=metric,
        interpolation=interpolation,
        fixed_mask=args.fixed_mask,
        moving_mask=args.moving_mask,
        verbose=args.verbose,
    )
    print(f"Registered S0: {registered}")

    # Get transform files
    transforms = get_transform_files(output_prefix, transform_type)
    print(f"Transforms: {[str(t) for t in transforms]}")

    # Apply transforms to additional images if specified
    if args.apply_to:
        print(f"\nApplying transforms to {len(args.apply_to)} additional image(s)...")
        for img in args.apply_to:
            if not img.exists():
                print(f"Warning: Image not found, skipping: {img}")
                continue

            output = args.output_dir / f"{img.stem}_registered.nii.gz"
            print(f"  Transforming: {img.name} -> {output.name}")

            apply_transforms(
                fixed=args.t1w,
                moving=img,
                output=output,
                transforms=transforms,
                interpolation=interpolation,
                verbose=args.verbose,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
