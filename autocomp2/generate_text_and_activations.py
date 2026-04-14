#!/usr/bin/env python3
"""Generate autocompletion text and activations for deception detection analysis.

For each conversation (system prompt + user prompt + reference answer) and each
prefix length i, this script:
  1. Feeds the model P^{(k)} + R^{(k)}_i and generates m tokens (autocompletion).
  2. Captures residual-stream activations at specified layers during generation.

Outputs are saved per model in the data/ subdirectory:
  - JSON file with text (continuous strings + token-by-token)
  - PT file with activation tensor of shape (N, c, m, L, d)
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ── Hyperparameters (adjust as needed) ───────────────────────────────────────
M = 10          # autocompletion length: number of tokens to generate per prefix
C = 15          # considered length: number of prefix positions to iterate over
JSONL_FILE = "autoconv3.jsonl"
MODELS_FILE = "models.md"
DATA_DIR = "data"


def get_default_layer_indices(num_layers):
    """Select layers at D//4, 3*D//8, D//2, 3*D//4 of the model's depth D."""
    D = num_layers
    return [D // 4, 3 * D // 8, D // 2, 3 * D // 4]


def load_models_list():
    with open(MODELS_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def load_conversations():
    convos = []
    with open(JSONL_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            msgs = json.loads(line.strip())
            sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
            usr_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            ref_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            convos.append({"system": sys_msg, "user": usr_msg, "reference": ref_msg})
    return convos


def model_id_to_filename(model_id):
    """Replace '/' with '--' for safe filenames."""
    return model_id.replace("/", "--")


def format_prompt(tokenizer, system_prompt, user_prompt):
    """Build the formatted prompt P^{(k)} using the model's chat template.

    Falls back to folding the system prompt into the user message for models
    that don't support a system role (e.g. Gemma 2).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return ids
    except Exception:
        pass

    combined = f"{system_prompt}\n\n{user_prompt}"
    messages = [{"role": "user", "content": combined}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return ids


def get_eos_token_ids(model, tokenizer):
    """Collect all EOS / end-of-turn token IDs for the model."""
    eos_ids = set()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        eid = model.generation_config.eos_token_id
        if eid is not None:
            if isinstance(eid, list):
                eos_ids.update(eid)
            else:
                eos_ids.add(eid)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    return eos_ids


# ── Activation capture via forward hooks ─────────────────────────────────────

def _find_decoder_layers(model):
    """Locate the nn.ModuleList of decoder layers across model architectures.

    Handles standard CausalLM (model.model.layers), multimodal models like
    Gemma 3 (model.language_model.model.layers), and GPT-style models
    (model.transformer.h).
    """
    candidates = [
        lambda m: m.model.layers,                        # Llama, Gemma 2, Qwen
        lambda m: m.language_model.model.layers,         # Gemma 3
        lambda m: m.model.language_model.model.layers,   # Gemma 3 alt
        lambda m: m.transformer.h,                       # GPT-style
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise ValueError(f"Cannot locate decoder layers in {type(model).__name__}")


def _get_hidden_size(model):
    """Get the residual-stream dimension, checking nested configs."""
    config = model.config
    if hasattr(config, "hidden_size") and config.hidden_size is not None:
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise ValueError(f"Cannot determine hidden_size for {type(model).__name__}")


class ActivationCapturer:
    """Register hooks on specified decoder layers to capture residual-stream
    activations (the output hidden states of each hooked layer)."""

    def __init__(self, model, layer_indices):
        self.layer_indices = layer_indices
        self.activations = {}
        self.hooks = []
        self.capture_all_positions = False  # True during prefill

        layers = _find_decoder_layers(model)
        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def fn(module, input, output):
            hidden = output[0].detach()
            # Some transformers versions return (seq_len, d) instead of
            # (batch, seq_len, d).  Normalise to 3-D.
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            if self.capture_all_positions:
                self.activations[layer_idx] = hidden.cpu()
            else:
                self.activations[layer_idx] = hidden[:, -1:, :].cpu()
        return fn

    def get_and_clear(self):
        acts = self.activations
        self.activations = {}
        return acts

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── KV-cache utilities ──────────────────────────────────────────────────────

def clone_kv_cache(cache, end_pos):
    """Return a deep copy of *cache* truncated to the first *end_pos* positions.

    Handles both newer transformers (key_cache/value_cache attributes) and
    older versions (tuple-of-tuples), as well as DynamicCache with only the
    .update() / len() interface.
    """
    # ── Tuple-of-tuples cache (older transformers) ───────────────────────
    if isinstance(cache, tuple):
        return tuple(
            (k[:, :, :end_pos, :].clone(), v[:, :, :end_pos, :].clone())
            for k, v in cache
        )

    # ── DynamicCache with key_cache / value_cache lists (transformers ≥4.38)
    if hasattr(cache, "key_cache"):
        new = DynamicCache()
        for layer_idx in range(len(cache.key_cache)):
            k = cache.key_cache[layer_idx][:, :, :end_pos, :].clone()
            v = cache.value_cache[layer_idx][:, :, :end_pos, :].clone()
            new.update(k, v, layer_idx)
        return new

    # ── Older DynamicCache that exposes items via __getitem__ / __len__ ──
    new = DynamicCache()
    for layer_idx in range(len(cache)):
        k, v = cache[layer_idx]
        new.update(
            k[:, :, :end_pos, :].clone(),
            v[:, :, :end_pos, :].clone(),
            layer_idx,
        )
    return new


# ── Core generation / activation-extraction logic ────────────────────────────

def process_model(model_id, conversations, m, c, layer_indices,
                  text_only=False, activation_only=False):
    """Run the full pipeline for one model: generate text and/or extract activations."""

    fname = model_id_to_filename(model_id)
    json_path = os.path.join(DATA_DIR, f"{fname}.json")
    pt_path = os.path.join(DATA_DIR, f"{fname}.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # ── activation-only mode: load existing JSON, recompute activations ──
    if activation_only:
        print(f"Loading existing JSON from {json_path} ...")
        with open(json_path) as f:
            saved_data = json.load(f)
        print(f"Loading model {model_id} ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        _run_activation_only(model, tokenizer, saved_data, layer_indices,
                             pt_path, device)
        del model
        torch.cuda.empty_cache()
        return

    # ── normal / text-only mode ──────────────────────────────────────────
    print(f"Loading model {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    eos_ids = get_eos_token_ids(model, tokenizer)
    capture_acts = not text_only
    capturer = ActivationCapturer(model, layer_indices) if capture_acts else None

    N = len(conversations)
    d = _get_hidden_size(model)
    L = len(layer_indices)
    if capture_acts:
        all_acts = torch.zeros(N, c, m, L, d, dtype=torch.bfloat16)

    json_data = {
        "model": model_id,
        "m": m,
        "c": c,
        "layers": layer_indices,
        "conversations": [],
    }

    for k, conv in enumerate(conversations):
        print(f"  [{k+1}/{N}] Conversation k={k} ...")

        prompt_ids = format_prompt(tokenizer, conv["system"], conv["user"])
        ref_ids = tokenizer.encode(conv["reference"], add_special_tokens=False)
        p = len(prompt_ids)
        n_k = len(ref_ids)
        c_k = min(c, n_k)

        # ── Prefill: single forward pass over P + R ──────────────────────
        # Thanks to causal attention, position p+i-1 in this pass gives us
        # the exact same hidden state / logits as if only P + R_i were fed.
        full_ids = prompt_ids + ref_ids
        input_tensor = torch.tensor([full_ids], device=device)

        if capturer is not None:
            capturer.capture_all_positions = True

        with torch.no_grad():
            output = model(input_tensor, use_cache=True)

        base_cache = output.past_key_values
        prefill_logits = output.logits[0].cpu()  # (seq_len, vocab_size)

        if capturer is not None:
            prefill_acts = capturer.get_and_clear()   # {layer: (1, seq_len, d)}
            capturer.capture_all_positions = False

        # ── Generate autocompletions for each prefix i ───────────────────
        conv_entry = {
            "k": k,
            "system_prompt": conv["system"],
            "user_prompt": conv["user"],
            "reference_answer": conv["reference"],
            "reference_tokens": [tokenizer.decode([tid]) for tid in ref_ids],
            "reference_token_ids": ref_ids,
            "n": n_k,
            "prompt_token_ids": prompt_ids,
            "autocompletions": [],
        }

        for i in range(c_k):
            pos = p + i - 1  # last-token position of P + R_i in the prefill

            # j = 0 activation (prefix activation) ───────────────────────
            if capture_acts:
                for l_idx, layer in enumerate(layer_indices):
                    all_acts[k, i, 0, l_idx] = prefill_acts[layer][0, pos]

            # First autocompletion token ──────────────────────────────────
            first_token = prefill_logits[pos].argmax().item()
            ac_token_ids = [first_token]
            eos_pos = 0 if first_token in eos_ids else None

            # Generate remaining m-1 tokens ───────────────────────────────
            if eos_pos is None:
                gen_cache = clone_kv_cache(base_cache, p + i)
                next_tok = torch.tensor([[first_token]], device=device)

                for j in range(1, m):
                    with torch.no_grad():
                        out = model(next_tok, past_key_values=gen_cache,
                                    use_cache=True)
                    gen_cache = out.past_key_values

                    if capture_acts:
                        j_acts = capturer.get_and_clear()
                        for l_idx, layer in enumerate(layer_indices):
                            all_acts[k, i, j, l_idx] = j_acts[layer][0, 0]

                    next_id = out.logits[0, -1].argmax().item()
                    ac_token_ids.append(next_id)

                    if next_id in eos_ids:
                        eos_pos = j
                        break

                    next_tok = torch.tensor([[next_id]], device=device)

                del gen_cache

            # ── Store results ────────────────────────────────────────────
            actual_len = len(ac_token_ids)
            ac_tokens_str = [tokenizer.decode([tid]) for tid in ac_token_ids]
            ac_text = tokenizer.decode(ac_token_ids)

            # Pad to length m
            ac_tokens_str += [""] * (m - actual_len)
            padded_ids = ac_token_ids + [None] * (m - actual_len)

            conv_entry["autocompletions"].append({
                "i": i,
                "text": ac_text,
                "tokens": ac_tokens_str,
                "token_ids": padded_ids,
                "eos_position": eos_pos,
            })

        json_data["conversations"].append(conv_entry)

        del base_cache, prefill_logits
        if capture_acts:
            del prefill_acts
        torch.cuda.empty_cache()

    # ── Save outputs ─────────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved text  -> {json_path}")

    if capture_acts:
        torch.save(all_acts, pt_path)
        print(f"  Saved acts  -> {pt_path}")
        capturer.remove()

    del model
    torch.cuda.empty_cache()


def _run_activation_only(model, tokenizer, saved_data, layer_indices,
                         pt_path, device):
    """Re-derive activations from previously saved autocompletion tokens.

    For each (k, i), we forward P + R_i + AC_i in one pass and extract
    the activations at positions p+i-1 .. p+i+actual_len-2 (for j=0..actual_len-1).
    """
    m = saved_data["m"]
    c = saved_data["c"]
    N = len(saved_data["conversations"])
    d = _get_hidden_size(model)
    L = len(layer_indices)

    capturer = ActivationCapturer(model, layer_indices)
    all_acts = torch.zeros(N, c, m, L, d, dtype=torch.bfloat16)

    for conv in saved_data["conversations"]:
        k = conv["k"]
        prompt_ids = conv["prompt_token_ids"]
        ref_ids = conv["reference_token_ids"]
        p = len(prompt_ids)
        n_k = conv["n"]
        c_k = min(c, n_k)

        print(f"  [{k+1}/{N}] Activations for conversation k={k} ...")

        for i in range(c_k):
            ac = conv["autocompletions"][i]
            ac_ids = [tid for tid in ac["token_ids"] if tid is not None]

            full_input = prompt_ids + ref_ids[:i] + ac_ids
            input_tensor = torch.tensor([full_input], device=device)

            capturer.capture_all_positions = True
            with torch.no_grad():
                model(input_tensor)
            acts = capturer.get_and_clear()

            for j in range(min(len(ac_ids), m)):
                act_pos = p + i + j - 1
                for l_idx, layer in enumerate(layer_indices):
                    all_acts[k, i, j, l_idx] = acts[layer][0, act_pos]

            torch.cuda.empty_cache()

    torch.save(all_acts, pt_path)
    print(f"  Saved acts  -> {pt_path}")
    capturer.remove()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate autocompletion text and/or activations.")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID. If omitted, run all models "
                             "from models.md.")
    parser.add_argument("--textonly", action="store_true",
                        help="Generate text only (skip activation capture).")
    parser.add_argument("--activationonly", action="store_true",
                        help="Extract activations from an existing JSON "
                             "(skip text generation).")
    args = parser.parse_args()

    if args.textonly and args.activationonly:
        print("Error: --textonly and --activationonly are mutually exclusive.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    conversations = load_conversations()
    models = [args.model] if args.model else load_models_list()

    from transformers import AutoConfig

    for model_id in models:
        print(f"\n{'=' * 60}")
        print(f"  {model_id}")
        print(f"{'=' * 60}")

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Some models (e.g. Gemma 3) use a composite config where
        # num_hidden_layers lives inside a sub-config (text_config).
        if hasattr(config, "num_hidden_layers"):
            num_layers = config.num_hidden_layers
        elif hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
            num_layers = config.text_config.num_hidden_layers
        else:
            raise ValueError(f"Cannot determine num_hidden_layers for {model_id}")
        layer_indices = get_default_layer_indices(num_layers)
        print(f"  Depth D={num_layers}, "
              f"hooking layers {layer_indices}")

        process_model(
            model_id, conversations, M, C, layer_indices,
            text_only=args.textonly,
            activation_only=args.activationonly,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
