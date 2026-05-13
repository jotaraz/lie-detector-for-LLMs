"""
Convert a YAML dataset (e.g. more-roleplay.yaml) into a JSONL file in the
autoconv* format (e.g. autoconv8.jsonl).

Each line of the JSONL is a JSON array of role/content dicts:
    [
        {"role": "system",    "content": <scenario>},
        {"role": "user",      "content": <question>},
        {"role": "assistant", "content": <completion or "">},
    ]

If an entry has both `honest_completion` and `deceptive_completion`, two lines
are emitted per entry (honest first, then deceptive). If only a single
`completion` is present, one line is emitted with that value. If no completion
field is present at all, a single line is emitted with an empty assistant
content -- the field still exists.

Usage:
    python yaml_to_jsonl.py INPUT.yaml OUTPUT.jsonl [--include-answer-prefix]
"""

import argparse
import json
import sys

import yaml


def strip_outer_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


def build_conversation(scenario: str, question: str, completion: str) -> list:
    return [
        {"role": "system",    "content": scenario},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": completion},
    ]


def completions_for_entry(entry: dict) -> list:
    """Return list of (label, completion) for the entry.

    If no completion fields exist, returns [(None, "")] so a single line is
    emitted with an empty assistant content.
    """
    pairs = []
    if "honest_completion" in entry or "deceptive_completion" in entry:
        if "honest_completion" in entry:
            pairs.append(("honest", str(entry["honest_completion"])))
        if "deceptive_completion" in entry:
            pairs.append(("deceptive", str(entry["deceptive_completion"])))
    elif "completion" in entry:
        pairs.append((None, str(entry["completion"])))
    else:
        pairs.append((None, ""))
    return pairs


def convert(input_yaml: str, output_jsonl: str, include_answer_prefix: bool = False) -> int:
    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of entries at top level of {input_yaml}")

    n_written = 0
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for entry in data:
            scenario = str(entry.get("scenario", "")).strip()
            question = str(entry.get("question", "")).strip()
            answer_prefix = str(entry.get("answer_prefix", "")).strip()

            for _label, raw in completions_for_entry(entry):
                completion = strip_outer_quotes(raw) if raw else ""
                if include_answer_prefix and answer_prefix and completion:
                    completion = f"{answer_prefix} {completion}"
                conv = build_conversation(scenario, question, completion)
                out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                n_written += 1

    return n_written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("input_yaml")
    parser.add_argument("output_jsonl")
    parser.add_argument(
        "--include-answer-prefix",
        action="store_true",
        help="Prepend answer_prefix (e.g. 'Student:') to assistant content when present.",
    )
    args = parser.parse_args(argv)

    n = convert(args.input_yaml, args.output_jsonl, args.include_answer_prefix)
    print(f"Wrote {n} conversations to {args.output_jsonl}")


if __name__ == "__main__":
    main()
