"""
Conversation Autocomplete Script
=================================
For each conversation in a JSONL file, and for each model in MODELS, this script:
  1. Tokenizes the full assistant response
  2. At positions 0, k, 2k, ... generates a model completion
  3. Saves results to one JSONL file per model, named after both the conversation
     file and the model (e.g., autoconv1_google_gemma-2-9b-it.jsonl)

Each result line is written immediately after processing a conversation,
enabling real-time review.

Output format (one JSON object per line):
  - conversation:       the original messages (list of role/content dicts)
  - n_assistant_tokens: number of tokens in the original assistant response
  - autocompletes:      list of completions at each prefix length, each with:
      - tokens_provided:    how many assistant tokens were given as prefix
      - assistant_prefix:   the actual text of those tokens (human-readable)
      - completion:         what the model generated from that prefix
"""

import json
import subprocess
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# CONFIGURATION — edit this section
# =============================================================================

MODELS = [
    "Qwen/Qwen2.5-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.3-70B-Instruct",
]

# Path to the JSONL file containing conversations.
# Each line must be a JSON array of {"role": ..., "content": ...} dicts,
# with the last message from the "assistant".
# Can also be passed as a command-line argument: python autocomplete.py autoconv1.jsonl
CONVERSATIONS_FILE = Path("autoconv3.jsonl")

# Step size: generate completions at assistant token positions 0, K, 2K, ...
K = 1
# Max autocompletes: if the assistant response is >= M tokens, only the first M
# autocompletes (at steps 0, K, 2K, ..., (M-1)*K) are performed.
M = 10

# Generation settings
MAX_NEW_TOKENS = 100       # max tokens the model generates per completion
TEMPERATURE    = 1.0       # set to 1.0 for unmodified sampling; use do_sample=False for greedy
DO_SAMPLE      = False     # True = sampling, False = greedy decoding

# Output directory for per-model JSONL files
OUTPUT_DIR = Path("autocomplete_results")

# Device: "cuda", "mps", or "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# HELPERS
# =============================================================================

def load_conversations(path: Path) -> list[list[dict]]:
    """Load conversations from a JSONL file. Each line is a JSON array."""
    conversations = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            assert isinstance(parsed, list), f"Line {lineno}: expected a JSON array"
            conversations.append(parsed)
    return conversations


def conv_file_stem(path: Path) -> str:
    """Return the stem of the conversations file (e.g. 'autoconv1' from 'autoconv1.jsonl')."""
    return path.stem


def model_id_to_tag(model_id: str) -> str:
    """Convert a model ID like 'google/gemma-2-9b-it' to 'google_gemma-2-9b-it'."""
    return model_id.replace("/", "_")


def output_filename(conv_path: Path, model_id: str) -> str:
    return f"{conv_file_stem(conv_path)}_{model_id_to_tag(model_id)}.jsonl"


def load_model(model_id: str):
    print(f"\n{'='*60}")
    print(f"Loading model: {model_id}")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


def merge_system_into_user(conversation: list[dict]) -> list[dict]:
    """
    Converts a conversation with a leading system message into one without,
    by prepending the system content to the first user message.
    Used as a fallback for models that don't support the system role.
    """
    result = []
    system_text = None
    for msg in conversation:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user" and system_text is not None:
            result.append({"role": "user", "content": f"{system_text}\n\n{msg['content']}"})
            system_text = None
        else:
            result.append(msg)
    return result


def get_prompt_prefix_ids(tokenizer, conversation: list[dict]) -> list[int]:
    """
    Returns input_ids for everything up to (and including) the assistant turn
    header, i.e. exactly where the model would start generating.

    Uses apply_chat_template with add_generation_prompt=True, which is the
    canonical model-agnostic way to get the correct assistant-turn prefix.
    Falls back to merging the system message into the first user message for
    models that don't support the system role (e.g. Gemma 2).
    """
    # Strip the last assistant message — we'll add its tokens manually
    user_turns = conversation[:-1]
    try:
        prompt_ids = tokenizer.apply_chat_template(
            user_turns,
            add_generation_prompt=True,   # adds the assistant role header
            tokenize=True,
            return_tensors=None,          # plain Python list
        )
    except Exception as e:
        if "system role not supported" in str(e).lower():
            user_turns = merge_system_into_user(user_turns)
            prompt_ids = tokenizer.apply_chat_template(
                user_turns,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
        else:
            raise
    return prompt_ids


def tokenize_assistant_content(tokenizer, content: str) -> list[int]:
    """
    Tokenizes the assistant's content WITHOUT any special tokens,
    so we can prepend exactly i of them to the prompt prefix.
    """
    return tokenizer.encode(content, add_special_tokens=False)


def get_eos_token_ids(tokenizer, model_id: str) -> list[int]:
    """
    Returns the list of token IDs that should stop generation.
    For Gemma models, <end_of_turn> (id 106) is not in all_special_tokens,
    so skip_special_tokens=True won't strip it. We add it as an explicit
    stop token so generation halts there and the token never appears in output.
    """
    base = tokenizer.eos_token_id
    eos_ids = list(base) if isinstance(base, list) else [base]
    if "gemma" in model_id.lower():
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id != tokenizer.unk_token_id and eot_id not in eos_ids:
            eos_ids.append(eot_id)
    return eos_ids


def generate(model, tokenizer, input_ids: list[int], model_id: str = "", debug: bool = False) -> str:
    tensor = torch.tensor([input_ids], dtype=torch.long).to(model.device)
    mask   = torch.ones_like(tensor)
    eos_ids = get_eos_token_ids(tokenizer, model_id)
    pad_id  = tokenizer.eos_token_id if not isinstance(tokenizer.eos_token_id, list) else tokenizer.eos_token_id[0]
    with torch.no_grad():
        output = model.generate(
            tensor,
            attention_mask=mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else 1.0,
            pad_token_id=pad_id,
            eos_token_id=eos_ids,
        )
    # Decode only the newly generated tokens
    new_tokens = output[0][len(input_ids):]
    if debug:
        print(f"    [DEBUG] eos_ids={eos_ids}, new_token_ids={new_tokens.tolist()}")
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# =============================================================================
# MAIN
# =============================================================================

def run(conversations_file: Path, model_ids: list[str] | None = None):
    OUTPUT_DIR.mkdir(exist_ok=True)

    conversations = load_conversations(conversations_file)
    print(f"Loaded {len(conversations)} conversations from {conversations_file}")

    for model_id in (model_ids if model_ids is not None else MODELS):
        tokenizer, model = load_model(model_id)

        out_name = output_filename(conversations_file, model_id)
        output_file = OUTPUT_DIR / out_name

        with output_file.open("w", encoding="utf-8") as f:

            for conv_idx, conversation in enumerate(conversations):
                print(f"\n--- Conversation {conv_idx} ---")

                assert conversation[-1]["role"] == "assistant", (
                    f"Last message in conversation {conv_idx} must be from 'assistant'."
                )

                assistant_content = conversation[-1]["content"]
                prefix_ids        = get_prompt_prefix_ids(tokenizer, conversation)
                assistant_ids     = tokenize_assistant_content(tokenizer, assistant_content)
                n_tokens          = len(assistant_ids)

                step_end = (n_tokens + 1) if n_tokens < M else M * K
                print(f"  Assistant response: {n_tokens} tokens")
                print(f"  Generating at steps: {list(range(0, step_end, K))}")

                autocompletes = []
                for step in range(0, step_end, K):
                    partial_ids      = prefix_ids + assistant_ids[:step]
                    assistant_prefix = tokenizer.decode(assistant_ids[:step], skip_special_tokens=True)
                    debug            = (conv_idx == 0 and step == 0)
                    completion       = generate(model, tokenizer, partial_ids, model_id, debug=debug)

                    print(f"  step={step:3d} | tokens_provided={step:3d}/{n_tokens} | "
                          f"completion[:60]: {completion[:60]!r}")

                    autocompletes.append({
                        "tokens_provided":   step,
                        "assistant_prefix":  assistant_prefix,
                        "completion":        completion,
                    })

                entry = {
                    "conversation":       conversation,
                    "n_assistant_tokens": n_tokens,
                    "autocompletes":      autocompletes,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()

        print(f"\nResults saved to {output_file.resolve()}")

        # Free VRAM before loading next model
        del model
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("conversations_file", nargs="?", type=Path, default=CONVERSATIONS_FILE)
    parser.add_argument("--model", type=str, default=None,
                        help="Run for a single model (used internally by subprocess launcher)")
    args = parser.parse_args()

    if args.model is not None:
        # Single-model mode: called as a subprocess for one model
        run(args.conversations_file, model_ids=[args.model])
    else:
        # Launcher mode: spawn one subprocess per model, sequentially
        for model_id in MODELS:
            print(f"\n{'='*60}")
            print(f"Launching subprocess for model: {model_id}")
            print(f"{'='*60}")
            cmd = [sys.executable, __file__, str(args.conversations_file), "--model", model_id]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"ERROR: subprocess for {model_id} exited with code {result.returncode}", file=sys.stderr)
