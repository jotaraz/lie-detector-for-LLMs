#!/usr/bin/env python3
"""Install all dependencies needed to run scripts like deception_detection/scripts/experiment.py."""

import shutil
import subprocess
import sys


def find_python() -> str:
    """Return path to a Python interpreter that meets requires-python (>=3.11)."""
    candidates = [sys.executable] + [
        shutil.which(name) or ""
        for name in ["python3.13", "python3.12", "python3.11"]
    ] + [
        "/opt/az/bin/python3.13",
        "/home/ubuntu/.local/share/uv/python/cpython-3.11-linux-aarch64-gnu/bin/python3.11",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            version = subprocess.check_output(
                [candidate, "-c", "import sys; print(sys.version_info[:2])"],
                text=True,
            ).strip()
            major, minor = eval(version)
            if (major, minor) >= (3, 11):
                subprocess.check_output(
                    [candidate, "-c", "import lzma"],
                    stderr=subprocess.DEVNULL,
                )
                return candidate
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            continue

    raise RuntimeError(
        "No Python >= 3.11 found. Please install Python 3.11+ and re-run this script."
    )


def main():
    python = find_python()
    print(f"Using Python interpreter: {python}")

    # Ensure pip and setuptools are new enough for PEP 660 editable installs.
    subprocess.check_call([
        python, "-m", "pip", "install", "--upgrade", "pip", "setuptools>=64"
    ])
    # Install the package in editable mode, which pulls in all dependencies from pyproject.toml.
    # Use --extra-index-url for PyTorch CPU/CUDA variants if needed (defaults to PyPI torch).
    subprocess.check_call([
        python, "-m", "pip", "install", "-e", ".[dev]"
    ])
    print("\nInstallation complete. You can now run:")
    print("  python deception_detection/scripts/experiment.py run")


if __name__ == "__main__":
    main()
