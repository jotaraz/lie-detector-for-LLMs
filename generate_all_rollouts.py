"""
Generate rollouts locally for all models that need them.
Skips datasets that already have rollouts registered in dataset_paths.json.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DATASET_PATHS_FILE = REPO_ROOT / "dataset_paths.json"

# Each model and the eval datasets it needs rollouts for
# Only include datasets that are actually in eval_data configs
ROLLOUT_JOBS = [
    ("qwen-32b",    "roleplaying__plain"),
    ("qwen-32b",    "insider_trading__upscale"),
    ("gemma3-12b",  "roleplaying__plain"),
    ("gemma3-12b",  "insider_trading__upscale"),
    ("gemma-9b",    "roleplaying__plain"),
    ("llama-8b",    "roleplaying__plain"),
    ("llama-8b",    "insider_trading__upscale"),
    ("llama-3.1-70b-base", "roleplaying__plain"),
    ("llama-3.1-70b-base", "insider_trading__upscale"),
    ("llama-3.1-70b-base", "insider_trading_doubledown__upscale"),
    ("llama-3.1-70b-base", "sandbagging_v2__wmdp_mmlu"),
    ("llama-3.1-70b-base", "alpaca__plain"),
]


def load_dataset_paths() -> dict:
    if DATASET_PATHS_FILE.exists():
        with open(DATASET_PATHS_FILE) as f:
            return json.load(f)
    return {}


def already_generated(model: str, dataset: str, dataset_paths: dict) -> bool:
    key = f"{dataset}__{model}"
    if key not in dataset_paths:
        return False
    path = REPO_ROOT / dataset_paths[key]
    return path.exists()


def generate(python: str, model: str, dataset: str) -> bool:
    import subprocess
    print(f"\n{'='*60}")
    print(f"Generating rollouts: {dataset} for {model}")
    print(f"{'='*60}")
    result = subprocess.run(
        [
            python,
            "deception_detection/scripts/generate_rollouts.py",
            "--dataset_partial_id", dataset,
            "--model_name", model,
            "--use_api=False",
            "--grade_rollouts=True",
        ],
        check=False,
    )
    if result.returncode != 0:
        print(f"WARNING: rollout generation failed for {dataset} / {model} (exit {result.returncode})")
        return False
    return True


def main(python: str) -> None:
    dataset_paths = load_dataset_paths()

    already_done = [
        (model, dataset)
        for model, dataset in ROLLOUT_JOBS
        if already_generated(model, dataset, dataset_paths)
    ]
    pending = [
        (model, dataset)
        for model, dataset in ROLLOUT_JOBS
        if not already_generated(model, dataset, dataset_paths)
    ]

    print(f"[rollouts] {len(ROLLOUT_JOBS)} total rollout jobs across all models.")
    if already_done:
        print(f"[rollouts] {len(already_done)} already generated, skipping:")
        for model, dataset in already_done:
            print(f"[rollouts]   SKIP  {dataset} / {model}")

    if not pending:
        print("[rollouts] All rollouts already generated, nothing to do.")
        return

    print(f"[rollouts] {len(pending)} job(s) to generate:")
    for model, dataset in pending:
        print(f"[rollouts]   TODO  {dataset} / {model}")

    for i, (model, dataset) in enumerate(pending, 1):
        print(f"\n[rollouts] [{i}/{len(pending)}] Generating: {dataset} for {model}")
        success = generate(python, model, dataset)
        if success:
            print(f"[rollouts] [{i}/{len(pending)}] Done: {dataset} / {model}")
        else:
            print(f"[rollouts] [{i}/{len(pending)}] FAILED: {dataset} / {model}, continuing with next job.")

    print(f"\n[rollouts] Rollout generation finished.")


if __name__ == "__main__":
    python = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    main(python)
