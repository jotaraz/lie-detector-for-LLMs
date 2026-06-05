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

# Supported Python range: >= MIN, < MAX.
# Upper bound matters because the pinned deps (e.g. matplotlib~=3.9.2) ship no
# wheels for Python 3.14+, so uv would try to build them from source and fail
# (freetype 2.6.1 does not compile under clang 17).
MIN_PY = (3, 11)
MAX_PY = (3, 14)


def python_version(candidate: str) -> tuple[int, int] | None:
    """Return (major, minor) for a Python executable, or None if it can't run."""
    try:
        version = subprocess.check_output(
            [candidate, "-c", "import sys; print(sys.version_info[:2])"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return eval(version)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def find_python() -> str | None:
    """Return path to a supported Python (MIN_PY <= v < MAX_PY) with lzma, or None."""
    import glob
    uv_pythons = sorted(glob.glob(
        os.path.expanduser("~/.local/share/uv/python/cpython-3.1[1-9]*/bin/python3.*")
    ), reverse=True)
    # Prefer explicitly-named supported versions over sys.executable, which may
    # be a too-new interpreter (e.g. 3.14) that passes the lower bound but lacks
    # prebuilt wheels for the pinned dependencies.
    candidates = [
        shutil.which(name) or ""
        for name in ["python3.13", "python3.12", "python3.11"]
    ] + uv_pythons + [
        "/opt/az/bin/python3.13",
        "/home/ubuntu/.local/share/uv/python/cpython-3.11-linux-aarch64-gnu/bin/python3.11",
        sys.executable,
    ]

    for candidate in candidates:
        if not candidate:
            continue
        version = python_version(candidate)
        if version is None or not (MIN_PY <= version < MAX_PY):
            continue
        try:
            subprocess.check_output(
                [candidate, "-c", "import lzma"],
                stderr=subprocess.DEVNULL,
            )
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


def install_python_via_apt() -> None:
    """Install python3.12 via uv (preferred) or apt as fallback."""
    print("[installthings] No suitable Python found. Installing python3.12...")

    # uv can download standalone CPython builds without needing a PPA
    uv = find_uv()
    if not uv:
        print("[installthings] Installing uv to bootstrap Python install...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
        uv = find_uv()

    if uv:
        print("[installthings] Installing python3.12 via uv python install...")
        try:
            subprocess.check_call([uv, "python", "install", "3.12"])
            return
        except subprocess.CalledProcessError:
            print("[installthings] uv python install failed, falling back to apt...")

    print("[installthings] Falling back to apt. Running apt-get update first...")
    subprocess.check_call(["sudo", "apt-get", "update", "-y"])
    subprocess.check_call([
        "sudo", "apt-get", "install", "-y",
        "python3.12", "python3.12-full", "python3.12-dev", "python3-pip",
    ])


# ---------------------------------------------------------------------------
# Step 2: create an isolated virtual environment
# ---------------------------------------------------------------------------

def create_venv(base_python: str) -> str:
    """Create .venv with the given base Python; return path to venv Python.

    If a venv already exists but its Python is outside the supported range, it is
    removed and recreated — otherwise a stale 3.14 venv would keep failing to
    install the pinned deps.
    """
    import shutil as _shutil
    venv_python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.isfile(venv_python):
        existing = python_version(venv_python)
        if existing is not None and MIN_PY <= existing < MAX_PY:
            print(f"[installthings] Virtual environment already exists at {VENV_DIR}")
            return venv_python
        print(
            f"[installthings] Existing venv uses unsupported Python {existing}; "
            f"recreating with {base_python} ..."
        )
        _shutil.rmtree(VENV_DIR)

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
# Step 4a: install insider-trading runtime deps not declared in pyproject
# ---------------------------------------------------------------------------

# insider-trading/run.py (and the generate/extract modules it imports) pull in a
# couple of packages that the project's pyproject does NOT declare:
#   - filelock : run.py does `from filelock import SoftFileLock`
#   - pyyaml   : generate.py does `import yaml`
# These usually arrive transitively via torch/transformers, but that is not
# guaranteed, so install them explicitly to make `run.py` reliably importable.
INSIDER_TRADING_EXTRAS = ["filelock", "pyyaml"]


def install_insider_trading_extras(uv: str, venv_python: str) -> None:
    print("[installthings] Installing insider-trading extras (filelock, pyyaml) ...")
    subprocess.check_call([
        uv, "pip", "install",
        "--python", venv_python,
        *INSIDER_TRADING_EXTRAS,
    ])


# ---------------------------------------------------------------------------
# Step 4b: install vLLM (required by insider-trading/run.py generation phase)
# ---------------------------------------------------------------------------

def install_vllm(uv: str, venv_python: str) -> None:
    """Install vLLM + flashinfer-python into the venv.

    Required by insider-trading/run.py for the `generate` phase.
    Large download (~GB of CUDA wheels). flashinfer-python is needed by vLLM
    specifically for Gemma-2 softcap logits.

    The venv was created with --without-pip, so `pip install vllm` from an
    activated shell would silently install to ~/.local/lib/python3.10/ via the
    system pip — uv with --python pins the right interpreter.
    """
    print("[installthings] Installing vLLM + flashinfer-python into venv ...")
    try:
        subprocess.check_call([
            uv, "pip", "install",
            "--python", venv_python,
            "vllm",
            "flashinfer-python",
        ])
    except subprocess.CalledProcessError as e:
        print(f"[installthings] WARNING: vLLM install failed ({e}).")
        print(f"  Retry with:  {uv} pip install --python {venv_python} vllm flashinfer-python")


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

    # 4a. Extra runtime deps for insider-trading/run.py not declared in pyproject
    install_insider_trading_extras(uv, venv_python)

    # 4b. vLLM for the insider-trading generation phase
    install_vllm(uv, venv_python)

    run_cmd = f"{venv_python} {os.path.join(PROJECT_DIR, 'runthings.py')}"
    print("\n[installthings] ============================================================")
    print("[installthings] Installation complete!")
    print("[installthings] To run the experiments, execute:")
    print(f"\n    {run_cmd}\n")
    print("[installthings] ============================================================")


if __name__ == "__main__":
    main()
