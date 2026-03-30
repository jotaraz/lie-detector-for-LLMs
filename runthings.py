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


python = find_python()
print(f"[runthings] Using Python interpreter: {python}")

# Generate any missing rollouts before running experiments
print(f"\n[runthings] {'='*60}")
print(f"[runthings] STEP 1: Generating missing rollouts")
print(f"[runthings] {'='*60}")
subprocess.run([python, "generate_all_rollouts.py", python], check=True)
print(f"[runthings] Rollout generation complete.")

models = [
            "_llama-8b_layer8",
            "_llama-8b_layer16",
            "_llama-8b_layer24",
        ]

print(f"\n[runthings] {'='*60}")
print(f"[runthings] STEP 2: Running experiments for {len(models)} model config(s)")
print(f"[runthings] {'='*60}")

failed = []
for i, model in enumerate(models, 1):
    config_name = f"repe{model}"
    print(f"\n[runthings] [{i}/{len(models)}] Starting experiment: {config_name}")
    try:
        subprocess.run(
            [python, "deception_detection/scripts/experiment.py", "run", config_name],
            check=True,
        )
        print(f"[runthings] [{i}/{len(models)}] Experiment {config_name} COMPLETE")
    except subprocess.CalledProcessError as e:
        print(f"[runthings] [{i}/{len(models)}] Experiment {config_name} FAILED (exit {e.returncode}), continuing...")
        failed.append(config_name)

print(f"\n[runthings] {'='*60}")
print(f"[runthings] All done. {len(models) - len(failed)}/{len(models)} experiments succeeded.")
if failed:
    print(f"[runthings] Failed: {', '.join(failed)}")