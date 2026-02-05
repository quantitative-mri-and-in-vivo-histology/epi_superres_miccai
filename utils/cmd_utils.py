"""Command-line utilities for running FSL/MRtrix commands."""

import subprocess
from pathlib import Path


def run_command(cmd: str | list[str], cwd: Path | None = None, verbose: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result.

    Parameters
    ----------
    cmd : str or list[str]
        Command to run (string or list of arguments)
    cwd : Path, optional
        Working directory for the command
    verbose : bool
        Print command before running

    Returns
    -------
    subprocess.CompletedProcess
        Result with stdout, stderr, returncode

    Raises
    ------
    subprocess.CalledProcessError
        If command returns non-zero exit code
    """
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False

    if verbose:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        print(f"Running: {cmd_str}")

    result = subprocess.run(
        cmd,
        shell=shell,
        cwd=cwd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        result.check_returncode()

    return result
