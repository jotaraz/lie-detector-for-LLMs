#!/usr/bin/env python3
"""Merge several score-summary.json files into one per-model table.

Edit SUMMARY_FILES below to list the score-summary.json files (from different
directories) you want to merge. The script builds a table with one row per
model and one column per ac-tag (ac0 = no ac-tag, plus ac1, ac3, ac5).

Each cell shows honest / deceptive counts, where the contribution from each
listed summary file is kept as a separate addend (in the order the files are
listed). For example, if gemma-2-9b-it has 12 honest / 0 deceptive for ac0 in
the first file and 22 honest / 1 deceptive in the second, its ac0 cell reads:

    12+22 / 0+1

The table is printed to stdout (no file is written).
"""

import json
import os
import re
from collections import defaultdict

# --- Configure here ----------------------------------------------------------
# List the score-summary.json files to merge, in the order you want their
# contributions to appear in each cell's addition expressions.
SUMMARY_FILES = [
    "data-autoconv3/score-summary.json",
    "data-autoconv4/score-summary.json",
]

# ac-tag columns, in display order. "ac0" means scored files with no ac-tag.
AC_TAGS = ["ac0", "ac1", "ac3", "ac5"]
# -----------------------------------------------------------------------------

# scored-[ac<N>-]<date>-<time>-<model>.json
SCORED_RE = re.compile(r"^scored-(?:(ac\d+)-)?\d{8}-\d{6}-(.+)\.json$")


def parse_scored_name(name):
    """Return (ac_tag, model_display) for a scored-*.json filename, or None."""
    m = SCORED_RE.match(name)
    if not m:
        return None
    ac_tag = m.group(1) or "ac0"
    model = m.group(2)
    # Drop an org prefix like "google--" -> "gemma-2-9b-it".
    if "--" in model:
        model = model.split("--")[-1]
    return ac_tag, model


def main():
    # addends[model][ac_tag] = list of (honest, deceptive), one per summary file
    addends = defaultdict(lambda: defaultdict(list))
    models = []  # preserve first-seen order

    for summary_path in SUMMARY_FILES:
        if not os.path.isfile(summary_path):
            raise SystemExit(f"Error: summary file not found: {summary_path}")
        with open(summary_path) as f:
            summary = json.load(f)

        # Sum within this one file, so duplicate scored files for the same
        # model+ac-tag collapse into a single addend per summary file.
        per_file = defaultdict(lambda: [0, 0])  # (model, ac_tag) -> [honest, decep]
        for fname, counts in summary.get("files", {}).items():
            parsed = parse_scored_name(fname)
            if parsed is None:
                continue
            ac_tag, model = parsed
            per_file[(model, ac_tag)][0] += counts.get("honest", 0)
            per_file[(model, ac_tag)][1] += counts.get("deceptive", 0)

        for (model, ac_tag), (honest, decep) in per_file.items():
            if model not in addends:
                models.append(model)
            addends[model][ac_tag].append((honest, decep))

    if not models:
        raise SystemExit("Error: no scored entries found in any summary file.")

    models.sort()

    def cell(model, ac_tag):
        pairs = addends[model].get(ac_tag, [])
        if not pairs:
            return "-"
        honest = "+".join(str(h) for h, _ in pairs)
        decep = "+".join(str(d) for _, d in pairs)
        return f"{honest} / {decep}"

    # Build the table (header + rows).
    header = ["model name"] + AC_TAGS
    rows = [[m] + [cell(m, ac) for ac in AC_TAGS] for m in models]

    # Column widths for aligned plain-text output.
    widths = [len(h) for h in header]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt(row):
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(row))

    print(f"Merged {len(SUMMARY_FILES)} summary file(s); cells show "
          f"honest / deceptive (addend per file, in listed order):\n")
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


if __name__ == "__main__":
    main()
