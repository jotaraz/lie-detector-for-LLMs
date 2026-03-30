#!/usr/bin/env python3
"""Install all dependencies needed to run runthings.py.

Run this with the system Python (any version):
    python installthings.py

It will:
1. Find or install a Python >= 3.11 with working lzma support.
2. Create a virtual environment at .venv/ (isolated from system packages).
3. Ensure uv is available (installs it via pip if needed).
4. Use uv to install the project with all dependencies into the venv.
5. Print the exact command to use for running runthings.py.
"""

import shutil
import subprocess
import sys
import os


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, ".venv")


# ---------------------------------------------------------------------------
# Step 1: find or install a suitable base Python
# ---------------------------------------------------------------------------

def find_python() -> str | None:
    """Return path to a Python >= 3.11 with working lzma, or None."""
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
                stderr=subprocess.DEVNULL,
            ).strip()
            major, minor = eval(version)
            if (major, minor) < (3, 11):
                continue
            subprocess.check_output(
                [candidate, "-c", "import lzma"],
                stderr=subprocess.DEVNULL,
            )
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            continue
    return None


def install_python_via_apt() -> None:
    """Install python3.12 + python3.12-full via apt."""
    print("[installthings] No suitable Python found. Installing python3.12 via apt...")
    print("[installthings] Running apt-get update first...")
    subprocess.check_call(["sudo", "apt-get", "update", "-y"])
    subprocess.check_call([
        "sudo", "apt-get", "install", "-y",
        "python3.12", "python3.12-full", "python3.12-dev", "python3-pip",
    ])


# ---------------------------------------------------------------------------
# Step 2: create an isolated virtual environment
# ---------------------------------------------------------------------------

def create_venv(base_python: str) -> str:
    """Create .venv with the given base Python; return path to venv Python."""
    venv_python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.isfile(venv_python):
        print(f"[installthings] Virtual environment already exists at {VENV_DIR}")
    else:
        print(f"[installthings] Creating virtual environment at {VENV_DIR} ...")
        # --without-pip: uv will manage packages; we avoid the pip bootstrap delay
        subprocess.check_call([base_python, "-m", "venv", "--without-pip", VENV_DIR])
        print(f"[installthings] Virtual environment created.")
    return venv_python


# ---------------------------------------------------------------------------
# Step 3: ensure uv is available
# ---------------------------------------------------------------------------

def find_uv() -> str | None:
    uv = shutil.which("uv")
    if uv:
        return uv
    for path in [
        os.path.expanduser("~/.local/bin/uv"),
        os.path.expanduser("~/.cargo/bin/uv"),
    ]:
        if os.path.isfile(path):
            return path
    return None


def ensure_uv() -> str:
    uv = find_uv()
    if uv:
        return uv
    print("[installthings] uv not found. Installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
    uv = find_uv()
    if not uv:
        raise RuntimeError("uv installation failed — cannot find uv binary after install.")
    return uv


# ---------------------------------------------------------------------------
# Step 4: install project dependencies into the venv using uv
# ---------------------------------------------------------------------------

def install_deps(uv: str, venv_python: str) -> None:
    print(f"[installthings] Installing project into venv with uv ...")
    subprocess.check_call([
        uv, "pip", "install",
        "--python", venv_python,
        "-e", os.path.join(PROJECT_DIR, ".[dev]"),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(PROJECT_DIR)

    # 1. Find a suitable base Python
    base_python = find_python()
    if base_python is None:
        install_python_via_apt()
        base_python = find_python()
        if base_python is None:
            raise RuntimeError(
                "python3.12 was installed but still cannot be found with working lzma. "
                "Please check the installation manually."
            )
    print(f"[installthings] Base Python: {base_python}")

    # 2. Create isolated venv
    venv_python = create_venv(base_python)

    # 3. Ensure uv
    uv = ensure_uv()
    print(f"[installthings] Using uv: {uv}")

    # 4. Install deps
    install_deps(uv, venv_python)

    run_cmd = f"{venv_python} {os.path.join(PROJECT_DIR, 'runthings.py')}"
    print("\n[installthings] ============================================================")
    print("[installthings] Installation complete!")
    print("[installthings] To run the experiments, execute:")
    print(f"\n    {run_cmd}\n")
    print("[installthings] ============================================================")


if __name__ == "__main__":
    main()
