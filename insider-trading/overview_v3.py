import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "v3"


def fmt_counter(counter: Counter, total: int, indent: str) -> str:
    lines = []
    for value, count in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        lines.append(f"{indent}{value}: {count} ({count / total:.0%})")
    return "\n".join(lines) if lines else f"{indent}(none)"


def main():
    json_files = sorted(DATA_DIR.glob("*-generations.json"))
    if not json_files:
        print(f"No JSON files found in {DATA_DIR}")
        return

    for path in json_files:
        with open(path) as f:
            items = json.load(f)

        model = path.name.removesuffix("-generations.json")
        buy_counts = Counter()
        report_counts = Counter()
        doubling_by_report = {}
        for item in items:
            md = item.get("metadata", {})
            buy = md.get("intended_buy_LING", "<absent>")
            buy_counts[buy] += 1
            if buy != "yes":
                continue
            report = md.get("report_label", "<absent>")
            report_counts[report] += 1
            doubling = md.get("doubling_down_label")
            if doubling is not None:
                doubling_by_report.setdefault(report, Counter())[doubling] += 1

        n = len(items)
        n_buy = buy_counts.get("yes", 0)
        print(f"=== {model} ({n} rollouts) ===")
        print("  intended_buy_LING:")
        print(fmt_counter(buy_counts, n, "    "))
        print(f"  report_label (of {n_buy} buys):")
        print(fmt_counter(report_counts, n_buy or 1, "    "))
        print("  doubling_down_label by report_label:")
        if not doubling_by_report:
            print("    (none)")
        for report, counter in sorted(doubling_by_report.items()):
            n_report = report_counts[report]
            print(f"    {report} ({n_report}):")
            print(fmt_counter(counter, n_report, "      "))
        print()


if __name__ == "__main__":
    main()
