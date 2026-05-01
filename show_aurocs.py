#!/usr/bin/env python3
"""
Print roleplaying__plain AUROC by layer for all Qwen-72B probe results.
Reads directly from scores.json — no model loading required.
"""
import json
import re
from pathlib import Path

import torch
import yaml

from deception_detection.metrics import get_auroc
from deception_detection.scores import Scores
from deception_detection.types import Label
from deception_detection.utils import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
DATASET_KEY = "roleplaying__plain"


def mean_scores(scores: Scores, label: Label) -> torch.Tensor:
    """Mean of per-token scores for each dialogue with the given label."""
    indices = [i for i, l in enumerate(scores.labels) if l == label]
    return torch.tensor([
        scores.scores[i][~torch.isnan(scores.scores[i])].mean().item()
        for i in indices
    ])


rows = []
for folder in sorted(RESULTS_DIR.glob("*qwen-72b*")):
    sf = folder / "scores.json"
    if not sf.exists():
        continue
    with open(sf) as f:
        raw = json.load(f)
    if DATASET_KEY not in raw:
        continue

    scores = Scores.from_dict(raw[DATASET_KEY])
    honest = mean_scores(scores, Label.HONEST)
    deceptive = mean_scores(scores, Label.DECEPTIVE)
    if len(honest) == 0 or len(deceptive) == 0:
        continue

    try:
        # Probe scores honest responses higher, so flip direction to get AUROC > 0.5
        auroc = 1.0 - get_auroc(honest, deceptive)
    except Exception as e:
        print(f"  SKIP {folder.name}: {e}")
        continue

    # Read layer from cfg.yaml
    layer = -1
    cfg_path = folder / "cfg.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        layers = cfg.get("detect_layers", [])
        if layers:
            layer = layers[0]

    variant = "plain_with_sys" if "plain_with_sys" in folder.name else "plain"
    rows.append((variant, layer, auroc, folder.name))

rows.sort()
print(f"\n{'Variant':<20} {'Layer':>5}  {'AUROC':>6}")
print("-" * 38)
for variant, layer, auroc, _ in rows:
    print(f"{variant:<20} {layer:>5}  {auroc:.4f}")
print()
