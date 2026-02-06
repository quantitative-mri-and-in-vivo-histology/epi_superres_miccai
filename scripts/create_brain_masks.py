"""Create brain mask using FSL BET.

This script generates brain masks using FSL's BET (Brain Extraction Tool)
with the -R flag for robust brain centre estimation.
"""

import argparse
from pathlib import Path

from utils.cmd_utils import run_command


def create_brain_mask(
    input_image: Path,
    output_mask: Path,
    fractional_intensity: float = 0.5,
    robust: bool = True,
    verbose: bool = False,
) -> Path:
    """Create brain mask using FSL BET.

    Parameters
    ----------
    input_image : Path
        Input image
    output_mask : Path
        Output brain mask path
    fractional_intensity : float
        Fractional intensity threshold (0-1, smaller = larger brain)
        Default: 0.5
    robust : bool
        Use robust brain centre estimation (-R flag)
        Default: True
    verbose : bool
        Print verbose output
        Default: False

    Returns
    -------
    Path
        Path to brain mask
    """
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    # BET automatically adds suffix, so we need to handle the naming
    # Use stem without _brain_mask suffix for BET output
    if output_mask.stem.endswith("_brain_mask"):
        base_output = output_mask.parent / output_mask.stem.replace("_brain_mask", "")
    else:
        base_output = output_mask.parent / output_mask.stem

    # Build BET command
    # BET will create: base_output_brain.nii.gz and base_output_brain_mask.nii.gz
    cmd = f"bet {input_image} {base_output} -m -f {fractional_intensity}"

    if robust:
        cmd += " -R"

    if verbose:
        print(f"Running BET...")
        print(f"Command: {cmd}")

    run_command(cmd)

    # BET creates the mask with _brain_mask suffix
    bet_mask_output = Path(f"{base_output}_brain_mask.nii.gz")

    # Move to desired output location if different
    if bet_mask_output != output_mask:
        bet_mask_output.rename(output_mask)

        # Also clean up the brain-extracted image if we don't need it
        bet_brain_output = Path(f"{base_output}_brain.nii.gz")
        if bet_brain_output.exists():
            bet_brain_output.unlink()

    if verbose:
        print(f"Saved brain mask: {output_mask}")

    return output_mask


def main():
    parser = argparse.ArgumentParser(
        description="Create brain mask using FSL BET"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output brain mask",
    )
    parser.add_argument(
        "--fractional-intensity",
        "-f",
        type=float,
        default=0.5,
        help="Fractional intensity threshold (0-1, smaller = larger brain). Default: 0.5",
    )
    parser.add_argument(
        "--no-robust",
        action="store_true",
        help="Disable robust brain centre estimation (don't use -R flag)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    print(f"Creating brain mask using FSL BET")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Fractional intensity: {args.fractional_intensity}")
    print(f"Robust mode: {not args.no_robust}")
    print()

    create_brain_mask(
        args.input,
        args.output,
        fractional_intensity=args.fractional_intensity,
        robust=not args.no_robust,
        verbose=args.verbose,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
