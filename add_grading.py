#!/usr/bin/env python3
"""
Re-grades roleplaying__plain responses for Qwen-72B using the OpenAI API,
then patches the corrected labels into every Qwen-72B result folder.

Run from the repo root with the venv active:
    python add_grading.py

Requires OPENAI_API_KEY in .env (loaded automatically).

Steps:
  1. Re-grade data/rollouts/roleplaying__plain__qwen-72b.json
  2. Derive the correctly-shuffled label list (matching scores.json ordering)
  3. Patch 'roleplaying__plain' labels in all results/*qwen-72b*/scores*.json
"""

import concurrent.futures
import json
from pathlib import Path

import dotenv
import numpy as np
from tqdm import tqdm

from deception_detection.data.roleplaying import RolePlayingDataset
from deception_detection.rollouts.rollout_dialogue import RolloutDialogue
from deception_detection.utils import REPO_ROOT

# Load OPENAI_API_KEY (and any other keys) from .env
dotenv.load_dotenv(REPO_ROOT / ".env")

ROLLOUT_PATH = REPO_ROOT / "data/rollouts/roleplaying__plain__qwen-72b.json"
RESULTS_DIR = REPO_ROOT / "results"
DATASET_KEY = "roleplaying__plain"


# ── Step 1: re-grade ──────────────────────────────────────────────────────────

def regrade_rollout_file() -> None:
    print(f"[1/3] Loading rollout file: {ROLLOUT_PATH}")
    with open(ROLLOUT_PATH) as f:
        raw = json.load(f)

    rollout_dialogues = [RolloutDialogue(**r) for r in raw["rollouts"]]
    print(f"      {len(rollout_dialogues)} dialogues loaded.")

    # Clear every previously-failed grading entry
    for rd in rollout_dialogues:
        rd.graded_deception = []
        rd.grader_reasoning = []
        rd.deception_percentage = None

    print("      Submitting grading requests to OpenAI gpt-4o (20 workers)…")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(
                rd.grade_all,
                RolePlayingDataset.get_grading_prompt,
                False,   # verbose
                False,   # mock
            )
            for rd in rollout_dialogues
        ]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                      desc="Grading"):
            pass
        for fut in futures:
            fut.result()  # re-raise any exceptions

    unknown_count = sum(
        1 for rd in rollout_dialogues if "unknown" in rd.graded_deception
    )
    if unknown_count:
        print(f"      WARNING: {unknown_count} dialogue(s) still have 'unknown' judgements.")

    # Write back, preserving all top-level fields (dataset_class, base_name, …)
    raw["rollouts"] = [rd.model_dump() for rd in rollout_dialogues]
    with open(ROLLOUT_PATH, "w") as f:
        json.dump(raw, f, default=lambda o: o.__dict__, indent=2)
    print(f"      Saved → {ROLLOUT_PATH}")


# ── Step 2: compute the shuffled label list ───────────────────────────────────

def get_shuffled_labels() -> list[str]:
    """
    Reproduces the exact label ordering written into scores.json.

    from_rollouts() builds labels in rollout-file order; then
    randomly_shuffle_dialogues() permutes them with np.random.seed(42).
    get_dataset() passes shuffle_upon_init=True for 'roleplaying__plain'
    (it doesn't start with 'repe'), so the shuffle is always applied.
    """
    print(f"[2/3] Deriving shuffled label list from updated rollout file…")
    with open(ROLLOUT_PATH) as f:
        raw = json.load(f)

    raw_labels: list[str] = []
    for item in raw["rollouts"]:
        for i in range(len(item["output_strs"])):
            label = RolePlayingDataset._get_label_from_judgement(item["graded_deception"][i])
            raw_labels.append(label.name)  # uppercase name, matching Scores.to_dict() format

    np.random.seed(42)
    perm = np.random.permutation(len(raw_labels))
    shuffled = [raw_labels[i] for i in perm]

    dist = {v: shuffled.count(v) for v in sorted(set(shuffled))}
    print(f"      {len(shuffled)} labels  |  distribution: {dist}")
    return shuffled


# ── Step 3: patch result folders ─────────────────────────────────────────────

def patch_result_folders(new_labels: list[str]) -> None:
    folders = sorted(RESULTS_DIR.glob("*qwen-72b*"))
    print(f"[3/3] Patching {len(folders)} Qwen-72B result folder(s)…")

    patched = skipped = 0
    for folder in folders:
        folder_updated = False
        for fname in ("scores.json", "scores_all_tokens.json"):
            path = folder / fname
            if not path.exists():
                continue
            with open(path) as f:
                scores = json.load(f)
            if DATASET_KEY not in scores:
                continue
            existing = scores[DATASET_KEY]["labels"]
            if len(existing) != len(new_labels):
                print(f"      SKIP {folder.name}/{fname}: "
                      f"length {len(existing)} ≠ {len(new_labels)}")
                skipped += 1
                continue
            scores[DATASET_KEY]["labels"] = new_labels
            with open(path, "w") as f:
                json.dump(scores, f)
            folder_updated = True

        if folder_updated:
            patched += 1

    print(f"      Patched {patched} folder(s), skipped {skipped} file(s).")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    regrade_rollout_file()
    new_labels = get_shuffled_labels()
    patch_result_folders(new_labels)

    print(
        "\nDone.  To regenerate plots for each folder run:\n"
        "  for d in results/*qwen-72b*/; do\n"
        "    .venv/bin/python deception_detection/scripts/experiment.py results \"$d\"\n"
        "  done"
    )
