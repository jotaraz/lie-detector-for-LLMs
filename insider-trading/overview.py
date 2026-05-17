import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def main():
    json_files = sorted(DATA_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {DATA_DIR}")
        return

    for path in json_files:
        with open(path) as f:
            items = json.load(f)

        made_trade_counts = Counter()
        report_label_counts = Counter()
        for item in items:
            md = item.get("metadata", {})
            made_trade = md.get("made_trade")
            made_trade_counts[made_trade] += 1
            if made_trade == "yes":
                report_label_counts[md.get("report_label")] += 1

        print(f"=== {path.name} ({len(items)} items) ===")
        print("  made_trade:")
        for value, count in sorted(made_trade_counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            print(f"    {value}: {count}")
        print("  report_label (for made_trade=yes):")
        if not report_label_counts:
            print("    (none)")
        else:
            for value, count in sorted(report_label_counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
                print(f"    {value}: {count}")
        print()


if __name__ == "__main__":
    main()
