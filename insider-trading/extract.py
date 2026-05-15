"""Phase 2: residual-stream activation extraction from saved transcripts.

For each rollout produced by `generate.py`, this script identifies the
model-generated assistant turns (trade, optionally report, optionally
doubling-down), runs a single forward pass per turn through the HF model
under hooks, and saves activations at every assistant token.

Activations are residual-stream outputs of each hooked decoder block, at
layer indices [0, min(max_layer, depth)).

Output: <output_dir>/<hf-id-with-slash-as-dashes>-activations.pt
  {
    "model_id": str,
    "layer_indices": list[int],
    "hidden_size": int,
    "rollouts": list[dict[turn_name, Tensor[T_assistant, L, d] (bfloat16, cpu)]],
  }
where turn_name ∈ {"trade", "report", "double_down"}; missing turns are absent.

Index of model-generated assistant messages within the transcript, for the
default scenario: 16 (trade), 18 (report), 20 (double down). Derived from the
transcript-length distribution observed in the existing generations (17 / 19 /
21 messages depending on how far the dialogue progressed).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

TURN_MSG_INDEX = {"trade": 16, "report": 18, "double_down": 20}


# ── Model loading & introspection (adapted from autocomp2) ───────────────────


def _load_model(model_id: str):
    """bf16 + device_map='auto'; falls back to ConditionalGeneration for
    multimodal-only architectures (Gemma 3 etc.)."""
    kwargs = dict(dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForConditionalGeneration

        model = AutoModelForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def _find_decoder_layers(model) -> nn.ModuleList:
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.text_model.layers,
        lambda m: m.text_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.transformer.h,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    best, best_len = None, 0
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > best_len:
            first = module[0]
            if any(
                kw in n
                for n, _ in first.named_modules()
                for kw in ("attn", "attention", "self_attn")
            ):
                best, best_len = module, len(module)
    if best is not None:
        return best
    raise ValueError(f"Cannot locate decoder layers in {type(model).__name__}")


def _get_hidden_size(model) -> int:
    cfg = model.config
    if getattr(cfg, "hidden_size", None) is not None:
        return cfg.hidden_size
    if hasattr(cfg, "text_config") and getattr(cfg.text_config, "hidden_size", None) is not None:
        return cfg.text_config.hidden_size
    raise ValueError(f"Cannot determine hidden_size for {type(model).__name__}")


def _apply_chat_template(tokenizer, messages, **kwargs) -> list[int]:
    result = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(result, list):
        return result
    if hasattr(result, "input_ids"):
        ids = result["input_ids"]
        return ids if isinstance(ids, list) else list(ids)
    return list(result)


def _safe_apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    """Apply chat template; if the model rejects a system role (e.g. Gemma 2),
    fold the system content into the first user message and retry."""
    try:
        return _apply_chat_template(
            tokenizer, messages, add_generation_prompt=add_generation_prompt, tokenize=True
        )
    except Exception:
        pass
    if messages and messages[0].get("role") == "system":
        sys_content = messages[0]["content"]
        rest = messages[1:]
        if rest and rest[0].get("role") == "user":
            folded = [
                {"role": "user", "content": f"{sys_content}\n\n{rest[0]['content']}"},
                *rest[1:],
            ]
        else:
            folded = [{"role": "user", "content": sys_content}, *rest]
        return _apply_chat_template(
            tokenizer, folded, add_generation_prompt=add_generation_prompt, tokenize=True
        )
    raise RuntimeError("Failed to apply chat template")


# ── Activation capture ───────────────────────────────────────────────────────


class _Capturer:
    """Forward hooks on a chosen subset of decoder layers; stores each layer's
    residual-stream output (post-block hidden states) for the most recent
    forward pass."""

    def __init__(self, model, layer_indices: list[int]):
        self.layer_indices = layer_indices
        self.cache: dict[int, torch.Tensor] = {}
        self.hooks = []
        layers = _find_decoder_layers(model)
        for idx in layer_indices:
            self.hooks.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, layer_idx: int):
        def fn(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.detach()
            if h.dim() == 2:
                h = h.unsqueeze(0)
            # [1, T, d] → cpu bf16
            self.cache[layer_idx] = h.to(torch.bfloat16).cpu()

        return fn

    def get_and_clear(self) -> dict[int, torch.Tensor]:
        out, self.cache = self.cache, {}
        return out

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


@torch.no_grad()
def _extract_one_rollout(model, tokenizer, transcript, layer_indices: list[int], device, capturer):
    """For each model-generated assistant turn present in this rollout, return
    a [T_assistant, L, d] activation tensor."""
    out: dict[str, torch.Tensor] = {}
    transcript_len = len(transcript)

    for turn_name, msg_idx in TURN_MSG_INDEX.items():
        if msg_idx >= transcript_len:
            continue

        prefix_msgs = transcript[:msg_idx]
        prefix_ids = _safe_apply_chat_template(tokenizer, prefix_msgs, add_generation_prompt=True)
        assistant_text = transcript[msg_idx]["content"]
        assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
        if len(assistant_ids) == 0:
            continue

        full_ids = list(prefix_ids) + list(assistant_ids)
        x = torch.tensor([full_ids], device=device)
        _ = model(x, use_cache=False)
        layer_cache = capturer.get_and_clear()  # {layer_idx -> [1, T_full, d]}

        # stack across layers in the user-requested order → [T_full, L, d]
        stacked = torch.stack([layer_cache[i][0] for i in layer_indices], dim=1)
        n_prefix = len(prefix_ids)
        n_assist = len(assistant_ids)
        out[turn_name] = stacked[n_prefix : n_prefix + n_assist].clone()

    return out


def model_id_to_filename(model_id: str) -> str:
    return model_id.replace("/", "--")


def extract_for_model(
    model_id: str,
    output_dir: Path,
    max_layer: int = 60,
    limit: int | None = None,
) -> Path:
    json_path = output_dir / f"{model_id_to_filename(model_id)}-generations.json"
    pt_path = output_dir / f"{model_id_to_filename(model_id)}-activations.pt"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing transcripts: {json_path}. Run phase=generate first (or phase=all)."
        )

    print(f"  loading transcripts: {json_path}")
    with open(json_path) as f:
        rollouts = json.load(f)
    if limit is not None:
        rollouts = rollouts[:limit]

    print(f"  loading tokenizer + model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = _load_model(model_id)
    device = next(model.parameters()).device

    num_layers = len(_find_decoder_layers(model))
    layer_indices = list(range(min(max_layer, num_layers)))
    d = _get_hidden_size(model)
    print(
        f"  depth={num_layers}, capturing layers={layer_indices[0]}..{layer_indices[-1]} "
        f"(L={len(layer_indices)}), hidden={d}"
    )

    capturer = _Capturer(model, layer_indices)
    results: list[dict[str, torch.Tensor]] = []
    try:
        for k, rollout in enumerate(rollouts):
            if k == 0 or (k + 1) % 50 == 0 or k == len(rollouts) - 1:
                print(f"    rollout {k + 1}/{len(rollouts)}")
            results.append(
                _extract_one_rollout(
                    model, tokenizer, rollout["transcript"], layer_indices, device, capturer
                )
            )
    finally:
        capturer.remove()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": model_id,
            "layer_indices": layer_indices,
            "hidden_size": d,
            "rollouts": results,
        },
        pt_path,
    )
    print(f"  wrote {pt_path}")

    del model
    torch.cuda.empty_cache()
    return pt_path
