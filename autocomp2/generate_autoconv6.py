"""
Generate autoconv6.jsonl from the first 100 scenarios in dataset.yaml.

Each scenario produces 2 conversation lines:
  1. honest completion
  2. deceptive completion

Format matches autoconv3/4/5: each line is a JSON array of role/content dicts.

Toggle:
  INCLUDE_ANSWER_PREFIX = True   →  prepend answer_prefix to assistant content
  INCLUDE_ANSWER_PREFIX = False  →  use completion text as-is
"""

import json
import yaml

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_YAML  = "../data/roleplaying/dataset.yaml"
OUTPUT_JSONL = "autoconv6.jsonl"
N_SCENARIOS = 100
INCLUDE_ANSWER_PREFIX = False   # toggle: True / False
# ─────────────────────────────────────────────────────────────────────────────


def strip_outer_quotes(text: str) -> str:
    """Remove a single layer of leading/trailing straight double-quotes."""
    text = text.strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    return text


def build_conversation(scenario: str, question: str, completion: str) -> list[dict]:
    return [
        {"role": "system",    "content": scenario},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": completion},
    ]


def main():
    with open(INPUT_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    scenarios = data[:N_SCENARIOS]

    lines = []
    for entry in scenarios:
        scenario         = entry["scenario"].strip()
        question         = entry["question"].strip()
        answer_prefix    = entry.get("answer_prefix", "").strip()
        honest_raw       = str(entry["honest_completion"]).strip()
        deceptive_raw    = str(entry["deceptive_completion"]).strip()

        honest    = strip_outer_quotes(honest_raw)
        deceptive = strip_outer_quotes(deceptive_raw)

        if INCLUDE_ANSWER_PREFIX and answer_prefix:
            honest    = f"{answer_prefix} {honest}"
            deceptive = f"{answer_prefix} {deceptive}"

        lines.append(build_conversation(scenario, question, honest))
        lines.append(build_conversation(scenario, question, deceptive))

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for conv in lines:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Wrote {len(lines)} conversations to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
