#!/usr/bin/env python3
"""Install all dependencies needed to run scripts like deception_detection/scripts/experiment.py."""

import subprocess
import sys


def main():
    # Install the package in editable mode, which pulls in all dependencies from pyproject.toml.
    # Use --extra-index-url for PyTorch CPU/CUDA variants if needed (defaults to PyPI torch).
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-e", ".[dev]"
    ])
    print("\nInstallation complete. You can now run:")
    print("  python deception_detection/scripts/experiment.py run")


if __name__ == "__main__":
    main()
