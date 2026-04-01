#!/usr/bin/env python3
"""
tryprobe.py — Apply trained probes to LLM-generated text and visualize probe scores.

Usage:
    python tryprobe.py \\
        --probes results/probe_dir1 results/probe_dir2 \\
        --conversations conversations.json \\
        [--max_new_tokens 200] \\
        [--output results.png] \\
        [--no_show]

Conversations file format (JSON):
    A list of conversations; each conversation is a list of messages with
    "role" (system / user / assistant) and "content" fields. A single
    conversation (list of messages) is also accepted.

    Example:
        [
          [{"role": "user", "content": "What is 2+2?"}],
          [
            {"role": "system", "content": "You are deceptive."},
            {"role": "user",   "content": "Is the sky blue?"}
          ]
        ]

All probe directories must reference the same LLM. The model is loaded once,
text is generated for every conversation, hidden-state activations are captured
token-by-token at the layers required by the probes, and every probe is scored
on every token of every conversation.  Results are shown as a per-conversation
line plot where each line is one probe.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from deception_detection.activations import Activations
from deception_detection.detectors import Detector, get_detector_class
from deception_detection.models import ModelName, get_model_and_tokenizer
from deception_detection.probe_architectures import get_probe_architecture_class
from deception_detection.scores import Scores


# ---------------------------------------------------------------------------
# Minimal stand-in for TokenizedDataset (only the fields detectors touch)
# ---------------------------------------------------------------------------

@dataclass
class _MinimalTokDataset:
    detection_mask: torch.Tensor   # [batch, seqpos] bool
    attention_mask: torch.Tensor   # [batch, seqpos] bool


def _wrap_activations(
    acts_tensor: torch.Tensor,  # [1, n_toks, n_layers, hidden_dim]
    layers: list[int],
) -> Activations:
    """Pack a raw activation tensor into an Activations object all detectors accept."""
    n_toks = acts_tensor.shape[1]
    mask = torch.ones(1, n_toks, dtype=torch.bool)
    tok_ds = _MinimalTokDataset(detection_mask=mask, attention_mask=mask)

    acts = Activations.__new__(Activations)
    acts.all_acts = acts_tensor.cpu()
    acts.tokenized_dataset = tok_ds   # type: ignore[assignment]
    acts.logits = None
    acts.layers = layers
    return acts


# ---------------------------------------------------------------------------
# Config / detector loading
# ---------------------------------------------------------------------------

def _load_cfg(probe_dir: Path) -> dict[str, Any]:
    cfg_path = probe_dir / "cfg.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg.yaml not found in {probe_dir}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_detector(probe_dir: Path, cfg: dict[str, Any]) -> Detector:
    detector_path = probe_dir / "detector.pt"
    if not detector_path.exists():
        raise FileNotFoundError(f"detector.pt not found in {probe_dir}")

    arch = cfg.get("probe_architecture")
    method = cfg.get("method")

    try:
        if arch:
            cls = get_probe_architecture_class(arch)
        elif method:
            cls = get_detector_class(method)
        else:
            raise ValueError("cfg.yaml has neither 'probe_architecture' nor 'method'")
        return cls.load(detector_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load detector from {probe_dir}: {exc}") from exc


# ---------------------------------------------------------------------------
# Generation + activation capture
# ---------------------------------------------------------------------------

def _generate_with_activations(
    model,
    tokenizer,
    conversation: list[dict[str, str]],
    layers: list[int],
    max_new_tokens: int,
) -> tuple[list[int], torch.Tensor]:
    """
    Autoregressively generate up to *max_new_tokens* tokens and capture hidden
    states at *layers* for every generated token.

    Returns
    -------
    generated_ids : list[int]
        Token IDs that were generated (excluding the prompt).
    acts_tensor : FloatTensor [1, n_generated, len(layers), hidden_dim]
    """
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    # Per-step accumulator — filled by the hooks
    _cur: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            _cur[layer_idx] = hidden[0, -1, :].detach().cpu()
        return hook

    hooks = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]

    generated_ids: list[int] = []
    step_acts: list[torch.Tensor] = []   # each entry: [n_layers, hidden_dim]
    past_kv = None

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                _cur.clear()
                ids = input_ids if step == 0 else torch.tensor([[generated_ids[-1]]], device=device)
                out = model(ids, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values

                next_id = int(out.logits[0, -1, :].argmax())
                generated_ids.append(next_id)
                # stack layers in the order given by *layers*
                step_acts.append(torch.stack([_cur[li] for li in layers], dim=0))

                if next_id == tokenizer.eos_token_id:
                    break
    finally:
        for h in hooks:
            h.remove()

    if not step_acts:
        raise RuntimeError("Model generated zero tokens.")

    # [n_steps, n_layers, hidden_dim] -> [1, n_steps, n_layers, hidden_dim]
    acts_tensor = torch.stack(step_acts, dim=0).unsqueeze(0)
    return generated_ids, acts_tensor


# ---------------------------------------------------------------------------
# Probe scoring — per-token trajectory
# ---------------------------------------------------------------------------

def _score_tokens(
    detector: Detector,
    acts_tensor: torch.Tensor,   # [1, n_toks, n_layers, hidden_dim]
    all_layers: list[int],
) -> np.ndarray:
    """
    Run *detector* on every generated token (all_acts=True mode).

    Returns a 1-D float32 numpy array of length n_toks.
    """
    acts = _wrap_activations(acts_tensor, all_layers)
    with torch.no_grad():
        scores_obj: Scores = detector.score(acts, all_acts=True)
    # scores_obj.scores is a list with one entry per dialogue; we have 1
    return scores_obj.scores[0].cpu().float().numpy()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _label(conv: list[dict[str, str]]) -> str:
    """Short label for a conversation: last user message."""
    user_msgs = [m["content"] for m in conv if m.get("role") == "user"]
    return user_msgs[-1] if user_msgs else "(no user message)"


def _trunc(s: str, n: int = 70) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _plot(
    probe_dirs: list[Path],
    conv_labels: list[str],
    token_strs_all: list[list[str]],
    scores_by_probe: list[list[np.ndarray]],
    output_path: Path,
) -> None:
    n_convs = len(conv_labels)
    n_probes = len(probe_dirs)
    colors = [plt.cm.tab10(i / max(n_probes, 1)) for i in range(n_probes)]

    fig = plt.figure(figsize=(14, max(4, 3.5 * n_convs) + 1.5))
    gs = gridspec.GridSpec(n_convs, 1, figure=fig, hspace=0.75)

    for ci in range(n_convs):
        ax = fig.add_subplot(gs[ci])
        tokens = token_strs_all[ci]
        n = len(tokens)
        x = np.arange(n)

        for pi, (pd, per_conv) in enumerate(zip(probe_dirs, scores_by_probe)):
            s = per_conv[ci][:n]
            ax.plot(x[: len(s)], s, label=_trunc(pd.name, 45), color=colors[pi], lw=1.5)

        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)

        # Sparse x-tick token labels
        step = max(1, n // 25)
        ticks = x[::step]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [tokens[t].replace("▁", " ").replace("Ġ", " ").replace("\n", "↵")[:8] for t in ticks],
            rotation=60, ha="right", fontsize=7,
        )

        ax.set_ylabel("Probe score", fontsize=8)
        ax.set_title(f"Conv {ci + 1}: {_trunc(conv_labels[ci])}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if n_probes > 1 or ci == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.75)

    fig.suptitle("Probe scores during generation", fontsize=12, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--probes", nargs="+", required=True, metavar="DIR",
        help="One or more probe result directories (each needs cfg.yaml + detector.pt).",
    )
    parser.add_argument(
        "--conversations", required=True, metavar="FILE|JSON",
        help=(
            "Path to a JSON file, or a raw JSON string. "
            "Format: list of conversations (or a single conversation)."
        ),
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200,
        help="Maximum tokens to generate per conversation (default: 200).",
    )
    parser.add_argument(
        "--output", default="tryprobe_results.png",
        help="Output PNG path (default: tryprobe_results.png).",
    )
    parser.add_argument(
        "--no_show", action="store_true",
        help="Skip plt.show() — just save.",
    )
    args = parser.parse_args()

    # ── Load probe configs ────────────────────────────────────────────────
    probe_dirs = [Path(p) for p in args.probes]
    configs: list[dict[str, Any]] = []
    for pd in probe_dirs:
        cfg = _load_cfg(pd)
        configs.append(cfg)
        arch = cfg.get("probe_architecture") or cfg.get("method")
        print(f"Probe  : {pd.name}")
        print(f"  architecture  : {arch}")
        print(f"  model_name    : {cfg.get('model_name')}")
        print(f"  detect_layers : {cfg.get('detect_layers')}")

    # ── Validate: all probes must use the same model ──────────────────────
    model_name_strs = [str(cfg.get("model_name")) for cfg in configs]
    if len(set(model_name_strs)) > 1:
        sys.exit(f"ERROR: probes target different models: {model_name_strs}")
    try:
        model_name = ModelName(model_name_strs[0])
    except ValueError:
        sys.exit(f"ERROR: unrecognised model name '{model_name_strs[0]}'")

    all_layers: list[int] = sorted(
        {layer for cfg in configs for layer in (cfg.get("detect_layers") or [])}
    )
    print(f"\nModel  : {model_name.value}")
    print(f"Layers : {all_layers}")

    # ── Load conversations ────────────────────────────────────────────────
    p = Path(args.conversations)
    if p.exists():
        raw = json.loads(p.read_text())
    else:
        try:
            raw = json.loads(args.conversations)
        except json.JSONDecodeError:
            sys.exit(f"ERROR: cannot parse conversations from '{args.conversations}'")

    # Accept a single conversation (list of message dicts) or a list of conversations
    if raw and isinstance(raw[0], dict):
        conversations: list[list[dict]] = [raw]
    else:
        conversations = raw

    conv_labels = [_label(c) for c in conversations]
    print(f"\nConversations : {len(conversations)}")
    for i, lbl in enumerate(conv_labels):
        print(f"  [{i + 1}] {_trunc(lbl, 80)}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model '{model_name.value}' …")
    model, tokenizer = get_model_and_tokenizer(model_name)
    model.eval()

    # ── Load detectors ────────────────────────────────────────────────────
    print("\nLoading detectors …")
    detectors: list[Detector] = []
    for pd, cfg in zip(probe_dirs, configs):
        det = _load_detector(pd, cfg)
        detectors.append(det)
        print(f"  {pd.name}: {type(det).__name__}")

    # ── Generate text and collect activations ─────────────────────────────
    print("\nGenerating text and capturing activations …")
    token_strs_all: list[list[str]] = []
    acts_per_conv: list[torch.Tensor] = []

    for ci, conv in enumerate(conversations):
        print(f"  [{ci + 1}/{len(conversations)}] ", end="", flush=True)
        gen_ids, acts = _generate_with_activations(
            model=model,
            tokenizer=tokenizer,
            conversation=conv,
            layers=all_layers,
            max_new_tokens=args.max_new_tokens,
        )
        token_strs = tokenizer.convert_ids_to_tokens(gen_ids) or []
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"{len(gen_ids)} tokens: {_trunc(generated_text, 80)!r}")

        token_strs_all.append(token_strs)
        acts_per_conv.append(acts)

    # ── Score each probe on each conversation ─────────────────────────────
    print("\nScoring probes …")
    scores_by_probe: list[list[np.ndarray]] = []
    for pi, det in enumerate(detectors):
        per_conv_scores: list[np.ndarray] = []
        for ci, acts in enumerate(acts_per_conv):
            s = _score_tokens(det, acts, all_layers)
            per_conv_scores.append(s)
        scores_by_probe.append(per_conv_scores)
        print(f"  {probe_dirs[pi].name}: scored {len(conversations)} conversation(s)")

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\nPlotting …")
    _plot(
        probe_dirs=probe_dirs,
        conv_labels=conv_labels,
        token_strs_all=token_strs_all,
        scores_by_probe=scores_by_probe,
        output_path=Path(args.output),
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
