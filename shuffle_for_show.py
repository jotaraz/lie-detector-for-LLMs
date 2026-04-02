#!/usr/bin/env python3
"""
shuffle_for_show.py — Match probe scores and labels back to their original rollout entries.

For each dataset in scores.json, loads the corresponding rollout file using the same
shuffle logic as the experiment, then combines probe scores + labels + rollout text
into a single JSON file.

Usage:
    python shuffle_for_show.py <probe_dir> [--output matched_rollouts.json]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from deception_detection.repository import DatasetRepository
from deception_detection.scores import Scores

REPOSITORY = DatasetRepository()


def load_dataset(dataset_name: str, model: str, trim_reasoning: bool, val_fraction: float,
                 train_data: str, train_model: str):
    """
    Load a dataset replicating the exact shuffle used by Experiment.

    Returns the loaded (and possibly sliced) DialogueDataset.

    Key rules (from experiment.py / data/base.py):
      - repe datasets are never shuffled (shuffle_upon_init=False)
      - all other datasets are shuffled with np.random.seed(42)
      - val split = last val_fraction of the (unshuffled) train dataset, no re-shuffle
      - trim_reasoning=False here so we get the full original response text;
        the permutation is unaffected because trim_reasoning never changes len(dataset)
    """
    is_val = dataset_name == f"{train_data}_val"

    if is_val:
        # Load full train dataset without shuffling (train_data starts with "repe")
        full_ds = REPOSITORY.get(
            train_data,
            model=train_model,
            trim_reasoning=False,
            shuffle_upon_init=False,
        )
        n_val = int(len(full_ds) * val_fraction)
        return full_ds[-n_val:]
    else:
        shuffle = not dataset_name.startswith("repe")
        return REPOSITORY.get(
            dataset_name,
            model=model,
            trim_reasoning=False,
            shuffle_upon_init=shuffle,
        )


def get_prompt_and_response(dialogue) -> tuple[list[dict], str]:
    """
    Split a dialogue into prompt messages and the response.

    Prompt: all messages with detect=False (the input scenario/conversation).
    Response: content of the last message with detect=True (the model's reply).
    """
    prompt = [{"role": m.role, "content": m.content} for m in dialogue if not m.detect]
    response = ""
    for msg in reversed(dialogue):
        if msg.detect:
            response = msg.content
            break
    return prompt, response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_dir", type=Path,
                        help="Path to a probe result directory (needs cfg.yaml + scores.json).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: <probe_dir>/matched_rollouts.json).")
    args = parser.parse_args()

    probe_dir: Path = args.probe_dir

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = probe_dir / "cfg.yaml"
    if not cfg_path.exists():
        sys.exit(f"ERROR: cfg.yaml not found in {probe_dir}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_name: str = cfg["model_name"]
    val_fraction: float = cfg.get("val_fraction", 0.2)
    train_data: str = cfg["train_data"]
    eval_model = model_name if cfg.get("eval_on_policy", True) else "prewritten"
    train_model = model_name if cfg.get("train_on_policy", False) else "prewritten"
    trim_reasoning: bool = cfg.get("trim_reasoning", True)

    print(f"Probe dir  : {probe_dir}")
    print(f"Model      : {model_name}")
    print(f"Train data : {train_data}  (val_fraction={val_fraction})")
    print(f"Eval model : {eval_model}")

    # ── Load scores ───────────────────────────────────────────────────────────
    scores_path = probe_dir / "scores.json"
    if not scores_path.exists():
        sys.exit(f"ERROR: scores.json not found in {probe_dir}")
    with open(scores_path) as f:
        raw_scores = json.load(f)
    all_scores = {name: Scores.from_dict(s) for name, s in raw_scores.items()}
    print(f"Datasets   : {list(all_scores.keys())}\n")

    # ── Match each dataset ────────────────────────────────────────────────────
    output: dict = {}

    for dataset_name, scores in all_scores.items():
        print(f"Processing {dataset_name} ({len(scores)} entries)...")

        dataset = load_dataset(
            dataset_name, eval_model, trim_reasoning, val_fraction, train_data, train_model
        )

        if len(dataset) != len(scores):
            print(f"  WARNING: length mismatch — dataset={len(dataset)}, scores={len(scores)}. "
                  "Skipping.")
            continue

        entries = []
        meta = dataset.metadata or {}

        for i in range(len(dataset)):
            score_values: list[float] = scores.scores[i].tolist()
            score_mean = sum(score_values) / len(score_values) if score_values else None

            prompt, response = get_prompt_and_response(dataset.dialogues[i])

            entry: dict = {
                "shuffled_index": i,
                "original_index": meta["id"][i] if "id" in meta else None,
                "label": scores.labels[i].value,
                "probe_score_mean": score_mean,
                "probe_scores": score_values,
                "prompt": prompt,
                "response": response,
                "graded_deception": meta["model_graded_deception"][i] if "model_graded_deception" in meta else None,
                "grader_reasoning": meta["model_grader_reasoning"][i] if "model_grader_reasoning" in meta else None,
                "expected_label": meta["expected_label"][i] if "expected_label" in meta else None,
            }
            entries.append(entry)

        output[dataset_name] = entries
        print(f"  done — {len(entries)} entries matched.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.output or probe_dir / "matched_rollouts.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
