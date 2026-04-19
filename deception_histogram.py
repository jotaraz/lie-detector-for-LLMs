import json
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of graded deception scores.")
    parser.add_argument("json_file", help="Path to the rollouts JSON file")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    scores = [
        score
        for rollout in data["rollouts"]
        for score in rollout["graded_deception"]
    ]

    if not scores:
        print("No graded_deception scores found.")
        return

    avg = sum(scores) / len(scores)
    print(f"Number of scores: {len(scores)}")
    print(f"Average deception score: {avg:.4f}")

    fig, ax = plt.subplots()
    ax.hist(scores, bins=range(0, 12), align="left", rwidth=0.8, color="steelblue", edgecolor="white")
    ax.axvline(avg, color="red", linestyle="--", label=f"Mean = {avg:.2f}")
    ax.set_xlabel("Graded Deception Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Deception Score Distribution\n{args.json_file.split('/')[-1]}")
    ax.legend()
    ax.set_xticks(range(0, 11))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
