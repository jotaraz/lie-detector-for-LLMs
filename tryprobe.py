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

def _get_decoder_layers(model) -> torch.nn.ModuleList:
    """
    Return the ModuleList of transformer decoder layers regardless of whether
    the model wraps the LM inside a vision/multimodal outer model.

    Tries common paths in order:
      1. model.model.layers            (most HF decoder-only models)
      2. model.model.language_model.layers  (Gemma3 / LLaVA-style VLMs)
    """
    inner = model.model
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return inner.language_model.layers
    raise AttributeError(
        f"Cannot locate decoder layers on {type(inner).__name__}. "
        "Add the correct path to _get_decoder_layers()."
    )


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

    hooks = [_get_decoder_layers(model)[li].register_forward_hook(make_hook(li)) for li in layers]

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


def _layout_tokens(
    tokens: list[str],
    scores: np.ndarray,
    chars_per_line: int,
) -> list[list[tuple[int, str, float]]]:
    """
    Wrap tokens into lines of at most *chars_per_line* characters.

    Returns a list of lines; each line is a list of (x_char, display_str, score).
    """
    lines: list[list[tuple[int, str, float]]] = []
    current_line: list[tuple[int, str, float]] = []
    x_char = 0

    for token, score in zip(tokens, scores):
        display = (
            token
            .replace("▁", " ")
            .replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<0x0A>", "\n")
            .replace("</s>", "")
        )
        parts = display.split("\n")
        for part_idx, part in enumerate(parts):
            if part_idx > 0:          # explicit newline in token → new line
                lines.append(current_line)
                current_line = []
                x_char = 0
            if not part:
                continue
            tw = len(part)
            if x_char > 0 and x_char + tw > chars_per_line:   # wrap
                lines.append(current_line)
                current_line = []
                x_char = 0
            current_line.append((x_char, part, float(score)))
            x_char += tw

    if current_line:
        lines.append(current_line)
    return lines


def _char_width_frac(font_size: int, axes_w_pts: float) -> tuple[float, int]:
    """
    Return (char_w_frac, chars_per_line) using exact font advance widths from
    TextPath — no heuristic multiplier needed.

    char_w_frac  : fraction of the axes width occupied by one monospace character
    chars_per_line : how many characters fit across the axes
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    fp = FontProperties(family="monospace", size=font_size)
    # Use a long string so rounding of a single glyph is negligible
    sample = "M" * 200
    tp = TextPath((0, 0), sample, prop=fp)
    char_w_pts = tp.get_extents().width / len(sample)

    cpl = max(20, int(axes_w_pts / char_w_pts))
    return char_w_pts / axes_w_pts, cpl


def _render_token_heatmap(
    ax,
    tokens: list[str],
    scores: np.ndarray,
    cmap,
    norm,
    font_size: int,
    char_w_frac: float,
    chars_per_line: int,
) -> int:
    """
    Render tokens as gapless highlighted text.

    Uses a blended transform: x in axes-fraction (so the measured char_w_frac
    maps exactly to one monospace character width), y in data units (so
    matplotlib handles vertical scaling automatically).  Rectangle patches and
    text objects share the same transform, guaranteeing zero gap between
    adjacent token boxes.
    """
    from matplotlib.transforms import blended_transform_factory

    lines = _layout_tokens(tokens, scores, chars_per_line)
    n_lines = len(lines)

    # y axis: one data unit per line, top line = n_lines-1 … bottom = 0
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, n_lines + 0.08)
    ax.set_facecolor("white")
    ax.axis("off")

    # x: axes fraction  |  y: data units
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    rect_pad  = 0.06   # fraction of a line height left blank above/below the rect
    rect_h    = 1.0 - 2 * rect_pad

    for li, line_tokens in enumerate(lines):
        y_bot    = (n_lines - 1 - li) + rect_pad   # data-unit bottom of rect
        y_center = y_bot + rect_h / 2

        for x_char, part, score in line_tokens:
            color = cmap(norm(score))
            x = x_char * char_w_frac
            w = len(part) * char_w_frac

            # Background — exact same transform as text, so no gap ever
            ax.add_patch(plt.Rectangle(
                (x, y_bot), w, rect_h,
                facecolor=color, edgecolor="none",
                transform=trans, clip_on=True, zorder=1,
            ))

            ax.text(
                x, y_center, part,
                ha="left", va="center",
                fontsize=font_size,
                fontfamily="monospace",
                transform=trans,
                color="black",
                clip_on=True,
                zorder=2,
            )

    return n_lines


def _plot(
    probe_dirs: list[Path],
    conv_labels: list[str],
    token_strs_all: list[list[str]],
    scores_by_probe: list[list[np.ndarray]],
    output_path: Path,
) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    n_convs = len(conv_labels)
    n_probes = len(probe_dirs)

    # ── Colormap ─────────────────────────────────────────────────────────────
    # Restrict to 20–80 % of RdBu_r so neither extreme goes near-black:
    #   left edge  → muted blue  (low / honest)
    #   centre     → white       (neutral)
    #   right edge → muted red   (high / deceptive)
    cmap = LinearSegmentedColormap.from_list(
        "RdBu_soft", plt.cm.RdBu_r(np.linspace(0.20, 0.80, 256))
    )
    all_scores = np.concatenate([s for per_conv in scores_by_probe for s in per_conv])
    abs_max = float(max(abs(all_scores.min()), abs(all_scores.max()), 1e-6))
    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)

    # ── Layout constants ──────────────────────────────────────────────────────
    fig_width   = 20          # inches — wide canvas for long lines
    font_size   = 11          # pt
    gs_left     = 0.01        # explicit gridspec margins → predictable axes width
    gs_right    = 0.99
    cbar_in     = 0.50
    suptitle_in = 0.35
    title_in    = 0.28        # inches per subplot title

    axes_w_pts = fig_width * (gs_right - gs_left) * 72   # pts
    char_w_frac, chars_per_line = _char_width_frac(font_size, axes_w_pts)

    # Line height: font_size * comfortable spacing, in inches
    line_h_in = font_size * 1.55 / 72

    # ── Per-row heights ───────────────────────────────────────────────────────
    n_rows = n_convs * n_probes
    row_heights: list[float] = []
    for ci in range(n_convs):
        for pi in range(n_probes):
            tokens = token_strs_all[ci]
            scores = scores_by_probe[pi][ci]
            n  = min(len(tokens), len(scores))
            nl = max(len(_layout_tokens(tokens[:n], scores[:n], chars_per_line)), 1)
            row_heights.append(nl * line_h_in + title_in)

    total_h = sum(row_heights) + cbar_in + suptitle_in

    fig = plt.figure(figsize=(fig_width, total_h))
    gs = gridspec.GridSpec(
        n_rows + 1, 1,
        figure=fig,
        height_ratios=row_heights + [cbar_in],
        hspace=0.45,
        left=gs_left,
        right=gs_right,
        top=1.0 - suptitle_in / total_h,
        bottom=cbar_in / total_h * 0.5,
    )

    ax_idx = 0
    for ci in range(n_convs):
        for pi in range(n_probes):
            ax = fig.add_subplot(gs[ax_idx])
            ax_idx += 1

            tokens = token_strs_all[ci]
            scores = scores_by_probe[pi][ci]
            n = min(len(tokens), len(scores))

            _render_token_heatmap(
                ax, tokens[:n], scores[:n], cmap, norm,
                font_size=font_size,
                char_w_frac=char_w_frac,
                chars_per_line=chars_per_line,
            )

            title = f"Conv {ci + 1} — {_trunc(conv_labels[ci], 110)}"
            if n_probes > 1:
                title += f"  ·  probe: {_trunc(probe_dirs[pi].name, 40)}"
            ax.set_title(title, fontsize=10, loc="left", pad=3, fontweight="bold")

    # ── Shared colorbar ───────────────────────────────────────────────────────
    cax = fig.add_subplot(gs[-1])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb  = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(
        "Probe score   (blue = honest  ·  white = neutral  ·  red = deceptive)",
        fontsize=10,
    )

    fig.suptitle("Token-level probe scores", fontsize=14, fontweight="bold",
                 y=1.0 - suptitle_in / total_h / 2)
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
        "--max_new_tokens", type=int, default=50,
        help="Maximum tokens to generate per conversation (default: 50).",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output PNG path.  "
            "Default: plots/<probe1>[+<probe2>…].png  (created automatically)."
        ),
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

    # ── Resolve output path ───────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        probe_tag = "+".join(pd.name for pd in probe_dirs)
        # Truncate to keep filenames sane on all filesystems
        if len(probe_tag) > 180:
            probe_tag = probe_tag[:180] + "…"
        output_path = plots_dir / f"{probe_tag}.png"

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\nPlotting …")
    _plot(
        probe_dirs=probe_dirs,
        conv_labels=conv_labels,
        token_strs_all=token_strs_all,
        scores_by_probe=scores_by_probe,
        output_path=output_path,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
