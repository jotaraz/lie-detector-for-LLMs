#!/usr/bin/env python3
"""
Extract hidden-state activations from autoconv2_<model>.jsonl files.

For each conversation and model, three kinds of activations are recorded at
layers L//4, L//2 and 3·L//4 (L = number of transformer blocks).  In every
case only the *last token position* of the current sequence is saved — either
the last position of a teacher-forced prefix or the single new-token position
during a KV-cache generation step.

Activation types
────────────────
(1) Prefilled
    The model reads the entire conversation: system prompt, user turn, and
    the full ground-truth assistant answer.  The activation at the last token
    is saved.  One vector per conversation.

(2) Prefix i  (i = 0 … 9)
    The model reads the prompt (system + user, with add_generation_prompt=True
    so the assistant-turn header is appended) plus the first i tokens of the
    ground-truth assistant answer.  When i = 0 the input ends with the
    assistant-turn header and contains no content tokens yet.  The activation
    at the last token position is saved for each i.
    Ten vectors per conversation.

(3) Generation  (x steps per prefix)
    Starting from the KV cache produced by the prefix-i forward pass, the
    model greedily generates x tokens one at a time.  The KV cache is reused
    at every step so the prefix context does not need to be re-encoded.  The
    activation at the single new-token position is saved for each generated
    token.  10·x vectors per conversation.

Total per conversation: 1 + 10 + 10·x vectors, each of shape [3, D].

Output format
─────────────
One .pt file per model, saved to autocomplete_results/activations_<tag>.pt.
Load with torch.load(..., weights_only=True).  The file is a plain dict:

  "prefilled"    Tensor[N, 3, D]         conv × layer × emb
  "prefix"       Tensor[N, 10, 3, D]     conv × i × layer × emb
  "generation"   Tensor[N, 10, x, 3, D]  conv × i × step × layer × emb
  "layers"       list[int]               absolute hidden_states indices probed
  "model_id"     str                     HuggingFace model ID
  "n_gen_tokens" int                     x (generation steps per prefix)

N = number of conversations (12 for autoconv2 data), D = hidden dimension.
All activation tensors are stored as float32 on CPU.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

AUTOCOMPLETE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Chat-template helpers  (adapted from autocomplete.py in the repo root)
# ---------------------------------------------------------------------------

def merge_system_into_user(conversation: list[dict]) -> list[dict]:
    """Prepend the system message content into the first user message.

    Gemma 2 (and some other models) raise an error when the chat template
    encounters a 'system' role.  This helper folds the system content into the
    first user message, separated by a double newline, so the model still sees
    the full context without requiring explicit system-role support.
    """
    result, pending_system = [], None
    for msg in conversation:
        if msg["role"] == "system":
            pending_system = msg["content"]
        elif msg["role"] == "user" and pending_system is not None:
            result.append({"role": "user", "content": f"{pending_system}\n\n{msg['content']}"})
            pending_system = None
        else:
            result.append(msg)
    return result


def get_prompt_prefix_ids(tokenizer, conversation: list[dict]) -> list[int]:
    """Tokenise the prompt up to and including the assistant-turn header.

    Returns a plain list of token IDs.  The sequence ends with the
    model-specific assistant-role marker added by add_generation_prompt=True
    (e.g. '<|im_start|>assistant\\n' for Llama / Qwen, '<start_of_turn>model\\n'
    for Gemma), placing the model exactly where it would begin generating.

    Falls back to merge_system_into_user() when the tokenizer's chat template
    does not support the 'system' role (e.g. Gemma 2).
    """
    user_turns = [m for m in conversation if m["role"] != "assistant"]
    try:
        return tokenizer.apply_chat_template(
            user_turns,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
    except Exception as e:
        if "system role not supported" in str(e).lower():
            return tokenizer.apply_chat_template(
                merge_system_into_user(user_turns),
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
        raise


def tokenize_assistant_content(tokenizer, text: str) -> list[int]:
    """Tokenise assistant answer text without surrounding special tokens.

    Using add_special_tokens=False ensures that only the raw content tokens
    are returned — no BOS, no role markers — so we can prepend exactly i of
    them to the already-tokenised prompt prefix.
    """
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

# Maps model_tag (derived from filename: autoconv2_<tag>.jsonl) to:
#   (hf_model_id, extra_load_kwargs, needs_gemma3_patch, use_autocast)
#
# needs_gemma3_patch: Gemma 3's text model does not accumulate hidden states
#   natively; patch_gemma3_hidden_states() must be called after loading.
# use_autocast: torch.cuda.amp.autocast is enabled for Llama models to match
#   the practice in deception_detection/activations.py.
MODEL_CONFIGS: dict[str, tuple[str, dict, bool, bool]] = {
    "Qwen_Qwen2.5-32B-Instruct": (
        "Qwen/Qwen2.5-32B-Instruct",
        {"torch_dtype": torch.bfloat16, "device_map": "auto"},
        False, False,
    ),
    "google_gemma-2-9b-it": (
        "google/gemma-2-9b-it",
        {"torch_dtype": torch.bfloat16, "device_map": "auto", "attn_implementation": "eager"},
        False, False,
    ),
    "google_gemma-3-12b-it": (
        "google/gemma-3-12b-it",
        {"torch_dtype": torch.bfloat16, "device_map": "auto", "attn_implementation": "eager"},
        True, False,
    ),
    "meta-llama_Llama-3.1-8B-Instruct": (
        "meta-llama/Llama-3.1-8B-Instruct",
        {"torch_dtype": torch.float16, "device_map": "auto"},
        False, True,
    ),
    "meta-llama_Llama-3.3-70B-Instruct": (
        "meta-llama/Llama-3.3-70B-Instruct",
        {"torch_dtype": torch.float16, "device_map": "auto"},
        False, True,
    ),
}


def patch_gemma3_hidden_states(model) -> None:
    """Enable output_hidden_states=True for Gemma 3.

    Gemma3TextModel.forward does not accumulate per-layer hidden states by
    default, so output_hidden_states=True has no effect without patching.
    This function registers forward hooks on every transformer layer block so
    their outputs are collected into a hidden_states tuple that matches the
    standard HuggingFace format expected by the rest of this script.

    The hook captures the first element of each layer's output tuple (the
    hidden-state tensor) and appends it to a list.  After the forward pass,
    the list is attached to the result as result.hidden_states, with None at
    index 0 as a placeholder for the (uncaptured) embedding-layer output.

    Adapted from deception_detection/models.py::_patch_gemma3_hidden_states.
    """
    text_model = model.model.language_model  # Gemma3TextModel
    _orig = text_model.forward

    def patched_forward(**kwargs):
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        if not output_hidden_states:
            return _orig(**kwargs)

        collected: list = []

        def hook_fn(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            collected.append(tensor)

        handles = [layer.register_forward_hook(hook_fn) for layer in text_model.layers]
        try:
            result = _orig(**kwargs)
        finally:
            for h in handles:
                h.remove()

        # Index 0 is None (embedding placeholder); indices 1..L are layer outputs.
        result.hidden_states = (None,) + tuple(collected)
        return result

    text_model.forward = patched_forward


def load_model_and_tokenizer(model_tag: str):
    """Load model and tokenizer for the given model tag."""
    hf_id, load_kwargs, needs_patch, use_autocast = MODEL_CONFIGS[model_tag]
    print(f"  Loading tokenizer: {hf_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        padding_side="left",
        cache_dir=Path.home() / ".cache/huggingface",
    )
    print(f"  Loading model weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        cache_dir=Path.home() / ".cache/huggingface",
        **load_kwargs,
    )
    model.eval()
    if needs_patch:
        print("  Applying Gemma 3 hidden-states patch...", flush=True)
        patch_gemma3_hidden_states(model)
    return model, tokenizer, hf_id, use_autocast


def get_layer_indices(model) -> list[int]:
    """Return the three hidden_states indices to probe: L//4, L//2, 3*L//4."""
    L = model.config.num_hidden_layers
    return [L // 4, L // 2, 3 * L // 4]


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_and_capture(
    model,
    input_ids: torch.Tensor,
    layer_indices: list[int],
    use_autocast: bool,
    return_cache: bool = False,
) -> tuple[torch.Tensor, any, int]:
    """Single forward pass; return last-token activations and optionally KV cache.

    Args:
        model:        loaded HuggingFace causal LM (eval mode)
        input_ids:    LongTensor[1, seq_len]
        layer_indices: hidden_states indices to probe (e.g. [8, 16, 24])
        use_autocast: enable torch.cuda.amp.autocast (used for Llama models)
        return_cache: if True, also return past_key_values for subsequent
                      generate_step() calls; set False when the cache is not
                      needed to avoid unnecessary memory allocation

    Returns:
        acts          Tensor[n_probed, D]  last-token activations, float32 CPU
        past_kv       KV cache or None (depending on return_cache)
        next_token_id argmax of last-position logits (int)
    """
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    with torch.cuda.amp.autocast(enabled=use_autocast):
        out = model(
            ids,
            attention_mask=torch.ones_like(ids),
            output_hidden_states=True,
            use_cache=return_cache,
        )
    acts = torch.stack(
        [out.hidden_states[li][0, -1, :].cpu().float() for li in layer_indices]
    )  # [n_probed, D]
    next_tok = int(out.logits[0, -1, :].argmax())
    past_kv = out.past_key_values if return_cache else None
    del out
    return acts, past_kv, next_tok


@torch.no_grad()
def generate_step(
    model,
    token_id: int,
    past_kv,
    layer_indices: list[int],
    use_autocast: bool,
) -> tuple[torch.Tensor, any, int]:
    """One KV-cache-based autoregressive step.

    Feeds a single new token to the model given the cached prefix context.
    Because only one new token is processed, hidden_states[li] has shape
    [1, 1, D] and we index [0, 0, :] to extract the activation vector.

    Args:
        model:        loaded HuggingFace causal LM (eval mode)
        token_id:     the token to feed in (selected from the previous step's logits)
        past_kv:      KV cache from the preceding forward_and_capture or generate_step call
        layer_indices: hidden_states indices to probe
        use_autocast: enable torch.cuda.amp.autocast (used for Llama models)

    Returns:
        acts          Tensor[n_probed, D]  activation at the new token, float32 CPU
        new_past_kv   updated KV cache (includes the new token)
        next_token_id argmax of logits for the following position (int)
    """
    device = next(model.parameters()).device
    tok = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with torch.cuda.amp.autocast(enabled=use_autocast):
        out = model(
            tok,
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
    acts = torch.stack(
        [out.hidden_states[li][0, 0, :].cpu().float() for li in layer_indices]
    )  # [n_probed, D]  — single new-token position
    next_tok = int(out.logits[0, -1, :].argmax())
    new_past_kv = out.past_key_values
    del out
    return acts, new_past_kv, next_tok


# ---------------------------------------------------------------------------
# Per-entry extraction
# ---------------------------------------------------------------------------

def process_entry(
    model,
    tokenizer,
    entry: dict,
    layer_indices: list[int],
    x: int,
    use_autocast: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract all three activation types for one conversation entry.

    Returns:
        prefilled_acts   Tensor[n_probed, D]
        prefix_acts      Tensor[10, n_probed, D]
        generation_acts  Tensor[10, x, n_probed, D]
    """
    conversation = entry["conversation"]
    assistant_content = next(m["content"] for m in conversation if m["role"] == "assistant")

    # Tokenise the prompt (system + user + assistant-turn header)
    prompt_ids: list[int] = get_prompt_prefix_ids(tokenizer, conversation)
    # Tokenise the assistant answer content without special tokens
    asst_ids: list[int] = tokenize_assistant_content(tokenizer, assistant_content)

    # ---- (1) Prefilled: full conversation including complete assistant answer ----
    full_ids = torch.tensor([prompt_ids + asst_ids], dtype=torch.long)
    prefilled_acts, _, _ = forward_and_capture(
        model, full_ids, layer_indices, use_autocast, return_cache=False
    )
    del full_ids

    prefix_acts_list: list[torch.Tensor] = []     # accumulates 10 × [n_probed, D]
    generation_acts_list: list[torch.Tensor] = [] # accumulates 10 × [x, n_probed, D]

    for i in range(10):
        # ---- (2) Prefix i: prompt + first i assistant content tokens ----
        # When i == 0 the input is exactly the prompt with add_generation_prompt=True
        # (the assistant-turn header), and no content tokens are included yet.
        prefix_ids = torch.tensor([prompt_ids + asst_ids[:i]], dtype=torch.long)
        p_acts, past_kv, first_gen_tok = forward_and_capture(
            model, prefix_ids, layer_indices, use_autocast, return_cache=True
        )
        del prefix_ids
        prefix_acts_list.append(p_acts)

        # ---- (3) Greedy autoregressive generation for x steps from prefix i ----
        # first_gen_tok is the argmax of the logits at the last prefix position,
        # i.e. the token the model would generate first.  We feed it back in and
        # capture its activation, then continue for x steps total.
        gen_steps: list[torch.Tensor] = []
        cur_tok = first_gen_tok
        cur_past_kv = past_kv
        del past_kv
        for _step in range(x):
            step_acts, cur_past_kv, cur_tok = generate_step(
                model, cur_tok, cur_past_kv, layer_indices, use_autocast
            )
            gen_steps.append(step_acts)
        del cur_past_kv

        generation_acts_list.append(torch.stack(gen_steps))  # [x, n_probed, D]

    prefix_acts = torch.stack(prefix_acts_list)        # [10, n_probed, D]
    generation_acts = torch.stack(generation_acts_list) # [10, x, n_probed, D]
    return prefilled_acts, prefix_acts, generation_acts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--x",
        type=int,
        default=5,
        help="Number of tokens to generate per prefix position (default: 5)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Restrict to specific model tags, e.g. google_gemma-2-9b-it (default: all)",
    )
    args = parser.parse_args()
    x: int = args.x

    files = sorted(AUTOCOMPLETE_DIR.glob("autoconv2_*.jsonl"))
    if not files:
        print("No autoconv2_*.jsonl files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s).  n_gen_tokens (x) = {x}", flush=True)

    for jsonl_path in files:
        model_tag = jsonl_path.stem.removeprefix("autoconv2_")

        if args.models and model_tag not in args.models:
            print(f"\nSkipping {model_tag} (not in --models list)", flush=True)
            continue

        if model_tag not in MODEL_CONFIGS:
            print(f"\nSkipping {model_tag}: no entry in MODEL_CONFIGS", flush=True)
            continue

        out_path = AUTOCOMPLETE_DIR / f"activations_{model_tag}.pt"
        if out_path.exists():
            print(f"\nSkipping {model_tag}: {out_path.name} already exists", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Model tag : {model_tag}", flush=True)
        print(f"Input     : {jsonl_path.name}", flush=True)
        print(f"Output    : {out_path.name}", flush=True)

        with jsonl_path.open() as f:
            entries = [json.loads(line) for line in f if line.strip()]
        print(f"Conversations: {len(entries)}", flush=True)

        model, tokenizer, hf_id, use_autocast = load_model_and_tokenizer(model_tag)
        layer_indices = get_layer_indices(model)
        L = model.config.num_hidden_layers
        print(f"L = {L}  →  probing layers {layer_indices}", flush=True)

        all_prefilled: list[torch.Tensor] = []
        all_prefix: list[torch.Tensor] = []
        all_generation: list[torch.Tensor] = []

        for idx, entry in enumerate(tqdm(entries, desc="Extracting")):
            print(f"  Entry {idx + 1}/{len(entries)}", flush=True)
            pf, pr, gen = process_entry(
                model, tokenizer, entry, layer_indices, x, use_autocast
            )
            all_prefilled.append(pf)    # [n_probed, D]
            all_prefix.append(pr)       # [10, n_probed, D]
            all_generation.append(gen)  # [10, x, n_probed, D]

        result = {
            "prefilled":    torch.stack(all_prefilled),    # [N, 3, D]
            "prefix":       torch.stack(all_prefix),       # [N, 10, 3, D]
            "generation":   torch.stack(all_generation),   # [N, 10, x, 3, D]
            "layers":       layer_indices,
            "model_id":     hf_id,
            "n_gen_tokens": x,
        }

        torch.save(result, out_path)
        print(f"Saved → {out_path}", flush=True)

        # Free GPU memory before loading the next model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
