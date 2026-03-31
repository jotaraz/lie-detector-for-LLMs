"""
showresults.py — Display per-model result tables from the results/ directory.

For each model, prints a table with columns:
  probe | layers | train_dataset | <eval_dataset_1 auroc> | <eval_dataset_2 auroc> | ...

Flags:
  --onlycomplete   Hide rows that are missing any eval dataset present in other rows
                   for the same model.
"""

import sys
import csv
import yaml
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"

RELEVANT_MODELS = [
    "llama-8b",
    "llama-70b-3.3",
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


def load_results(exp_dir: Path) -> dict[str, tuple[float, float | None]]:
    """Return {dataset_name: (auroc, recall_1pct)} from results_table.csv."""
    csv_path = exp_dir / "results_table.csv"
    if not csv_path.exists():
        return {}
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


def fmt_layers(layers) -> str:
    if not layers:
        return "?"
    return "[" + ",".join(str(l) for l in layers) + "]"


def fmt_cell(value: tuple[float, float | None] | None) -> str:
    if value is None:
        return "      —      "
    auroc, recall = value
    recall_str = f"{recall:.2f}" if recall is not None else "  —  "
    return f"{auroc:.4f} ({recall_str})"


def print_model_table(model: str, rows: list[dict], all_datasets: list[str]) -> None:
    col_probe    = max(len("probe"),    max(len(r["probe"])    for r in rows))
    col_layers   = max(len("layers"),   max(len(r["layers"])   for r in rows))
    col_train    = max(len("train_data"), max(len(r["train_data"]) for r in rows))
    col_ds       = max(len("auroc (recall@1%)"), 17)

    header = (
        f"{'probe':<{col_probe}}  "
        f"{'layers':<{col_layers}}  "
        f"{'train_data':<{col_train}}  "
        + "  ".join(f"{ds:<{col_ds}}" for ds in all_datasets)
    )
    sep = "-" * len(header)

    print(f"\n{'═' * len(header)}")
    print(f"  Model: {model}")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    for r in sorted(rows, key=lambda x: (x["probe"], x["train_data"], x["layers"])):
        line = (
            f"{r['probe']:<{col_probe}}  "
            f"{r['layers']:<{col_layers}}  "
            f"{r['train_data']:<{col_train}}  "
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

        probe  = f"{cfg.get('id', '?')}_{cfg.get('method', '?')}"
        layers = fmt_layers(cfg.get("detect_layers"))
        train  = cfg.get("train_data", "?")
        aurocs = {
            ds: v
            for ds, v in load_results(exp_dir).items()
            if ds not in EXCLUDE_DATASETS
        }  # values are (auroc, recall_1pct) tuples

        model_data[model].append({
            "probe":      probe,
            "layers":     layers,
            "train_data": train,
            "aurocs":     aurocs,
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
