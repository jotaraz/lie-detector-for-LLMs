"""
showresults.py — Display per-model result tables from the results/ directory.

For each model, prints a table with columns:
  probe_type | probe | layers | train_dataset | <eval_dataset_1 auroc> | ...

probe_type distinguishes classic methods (lr, mms, lat, …) from new probe
architectures (mlp_paper, attention, ema, multimax, rolling_means_attention).

Results are loaded from results_table.csv when present (classic experiments),
or computed on the fly from scores.json for newer probe-architecture runs.

Flags:
  --onlycomplete   Hide rows that are missing any eval dataset present in other
                   rows for the same model.
"""

import json
import sys
import csv
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"

RELEVANT_MODELS = [
    "llama-8b",
    "llama-70b-3.3",
    "llama-70b",
    "gemma-9b",
    "gemma3-12b",
    "qwen-32b",
]

# Datasets to exclude from the AUROC columns (training/validation sets we don't
# care about in this overview).
EXCLUDE_DATASETS = set()

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_cfg(exp_dir: Path) -> dict | None:
    cfg_path = exp_dir / "cfg.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _auroc_and_recall_from_scores(scores_raw: list[list[float]], labels_raw: list[str]) -> tuple[float, float | None]:
    """Compute AUROC and recall@1%FPR from raw scores.json data for one dataset."""
    # Each dialogue score is a list; use the mean as the dialogue-level score.
    # Probes are trained so that honest > deceptive, so negate for detection.
    # Empty score lists (no detected tokens) map to 0.0.
    y_scores = np.array([-float(np.mean(s)) if len(s) > 0 else 0.0 for s in scores_raw])
    y_true = np.array([1 if lbl == "DECEPTIVE" else 0 for lbl in labels_raw])

    n_pos = y_true.sum()
    n_neg = (1 - y_true).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan"), None

    auroc = float(roc_auc_score(y_true, y_scores))

    # recall @ 1% FPR
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = int(np.searchsorted(fpr, 0.01))
    recall = float(tpr[min(idx, len(tpr) - 1)]) if len(tpr) > 0 else None
    return auroc, recall


def load_results(exp_dir: Path) -> dict[str, tuple[float, float | None]]:
    """Return {dataset_name: (auroc, recall_1pct)}.

    Tries results_table.csv first; falls back to computing from scores.json.
    """
    csv_path = exp_dir / "results_table.csv"
    if csv_path.exists():
        results = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row.get("") or row.get(None)  # index column (unnamed)
                if dataset and "auroc" in row:
                    try:
                        auroc = float(row["auroc"])
                    except (ValueError, TypeError):
                        continue
                    try:
                        recall = float(row["recall 1.0%"])
                    except (ValueError, TypeError, KeyError):
                        recall = None
                    results[dataset] = (auroc, recall)
        return results

    # Fallback: compute from scores.json
    scores_path = exp_dir / "scores.json"
    if not scores_path.exists():
        return {}
    with open(scores_path) as f:
        raw = json.load(f)
    results = {}
    for dataset, data in raw.items():
        try:
            auroc, recall = _auroc_and_recall_from_scores(data["scores"], data["labels"])
            results[dataset] = (auroc, recall)
        except Exception:
            continue
    return results


def probe_label(cfg: dict) -> tuple[str, str]:
    """Return (probe_type, probe_name) for display."""
    arch = cfg.get("probe_architecture")
    method = cfg.get("method")
    exp_id = cfg.get("id", "")

    if arch:
        probe_type = "probe_arch"
        name = f"{exp_id}_{arch}" if exp_id else arch
    else:
        probe_type = "method"
        name = f"{exp_id}_{method}" if exp_id else (method or "?")
    return probe_type, name


def fmt_layers(layers) -> str:
    if not layers:
        return "?"
    return "[" + ",".join(str(l) for l in layers) + "]"


def fmt_cell(value: tuple[float, float | None] | None) -> str:
    if value is None:
        return "      —      "
    auroc, recall = value
    if auroc != auroc:  # NaN check
        return "   NaN       "
    recall_str = f"{recall:.2f}" if recall is not None else "  —  "
    return f"{auroc:.4f} ({recall_str})"


def print_model_table(model: str, rows: list[dict], all_datasets: list[str]) -> None:
    col_type  = max(len("type"),       max(len(r["probe_type"]) for r in rows))
    col_probe = max(len("probe"),      max(len(r["probe"])      for r in rows))
    col_layers = max(len("layers"),    max(len(r["layers"])     for r in rows))
    col_train  = max(len("train_data"), max(len(r["train_data"]) for r in rows))
    col_dir    = max(len("dir"),        max(len(r["dir"])        for r in rows))
    col_ds     = max(len("auroc (recall@1%)"), 17)

    header = (
        f"{'type':<{col_type}}  "
        f"{'probe':<{col_probe}}  "
        f"{'layers':<{col_layers}}  "
        f"{'train_data':<{col_train}}  "
        f"{'dir':<{col_dir}}  "
        + "  ".join(f"{ds:<{col_ds}}" for ds in all_datasets)
    )
    sep = "-" * len(header)

    print(f"\n{'═' * len(header)}")
    print(f"  Model: {model}")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    for r in sorted(rows, key=lambda x: (x["probe_type"], x["probe"], x["train_data"], x["layers"])):
        line = (
            f"{r['probe_type']:<{col_type}}  "
            f"{r['probe']:<{col_probe}}  "
            f"{r['layers']:<{col_layers}}  "
            f"{r['train_data']:<{col_train}}  "
            f"{r['dir']:<{col_dir}}  "
            + "  ".join(
                f"{fmt_cell(r['aurocs'].get(ds)):<{col_ds}}"
                for ds in all_datasets
            )
        )
        print(line)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    only_complete = "--onlycomplete" in sys.argv
    # Collect data keyed by model name
    model_data: dict[str, list[dict]] = defaultdict(list)
    model_datasets: dict[str, set[str]] = defaultdict(set)

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        cfg = load_cfg(exp_dir)
        if cfg is None:
            continue

        model = cfg.get("model_name", "unknown")
        if model not in RELEVANT_MODELS:
            continue

        probe_type, probe = probe_label(cfg)
        layers = fmt_layers(cfg.get("detect_layers"))
        train  = cfg.get("train_data", "?")
        aurocs = {
            ds: v
            for ds, v in load_results(exp_dir).items()
            if ds not in EXCLUDE_DATASETS
        }  # values are (auroc, recall_1pct) tuples

        model_data[model].append({
            "probe_type": probe_type,
            "probe":      probe,
            "layers":     layers,
            "train_data": train,
            "aurocs":     aurocs,
            "dir":        exp_dir.name,
        })
        model_datasets[model].update(aurocs.keys())

    if not model_data:
        print(f"No results found in {RESULTS_DIR}")
        return

    for model in RELEVANT_MODELS:
        rows = model_data.get(model)
        if not rows:
            continue
        # Stable column order: sort datasets alphabetically, val sets last
        datasets = sorted(
            model_datasets[model],
            key=lambda d: (d.endswith("_val"), d),
        )
        if only_complete:
            full_set = set(datasets)
            rows = [r for r in rows if full_set.issubset(r["aurocs"].keys())]
            if not rows:
                continue
        print_model_table(model, rows, datasets)

    print()


if __name__ == "__main__":
    main()
