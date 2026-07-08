"""Push large result files to a HuggingFace dataset repo.

Large files (per-token score JSONs) are excluded from git and stored on HF instead.
Run this after an experiment to archive the large outputs.

Usage:
    python push_to_hf.py                          # push new/changed files only
    python push_to_hf.py --dry_run                # preview what would be uploaded
    python push_to_hf.py --repo_id other/repo     # override repo

Uploads use huggingface_hub.upload_large_folder, which handles rate limiting and
is resumable: if the process is interrupted (rate limit, Ctrl-C, crash), just run
it again and it continues where it left off.
"""

from pathlib import Path

import fire
from huggingface_hub import HfApi, create_repo

# Resolve everything relative to this script's location, NOT the current working
# directory — so the script works no matter where it is invoked from.
REPO_ROOT = Path(__file__).resolve().parent

# Set this to your HuggingFace repo (will be created if it doesn't exist)
REPO_ID = "jo-chen/lie-detector-results"

# These are the (dir, pattern) pairs excluded from git and stored on HF
LARGE_FILE_SPECS = [
    ("results", "*_all_tokens.json"),
    ("results", "control_scores.json"),
    ("results/probe-comparison", "*.pt"),
    ("results/probe-comparison", "*.json"),
    ("autocomplete_results", "*.pt"),
    ("autocomp2/data-autoconv3", "*.pt"),
    ("autocomp2/data-autoconv4", "*.pt"),
    ("autocomp2/data-autoconv5", "*.pt"),
    ("autocomp2/data-autoconv6", "*.pt"),
    ("autocomp2/data-autoconv7", "*.pt"),
    ("autocomp2/data-autoconv8", "*.pt"),
    ("autocomp2/data-autoconv9", "*.pt"),
    ("insider-trading/data", "*.pt"),
]


def push(
    repo_id: str = REPO_ID,
    dry_run: bool = False,
) -> None:
    files_to_upload: list[Path] = []
    print(f"Current working directory: {Path.cwd()}")
    print(f"Script/repo root (paths resolved against this): {REPO_ROOT}")
    for dir_name, pattern in LARGE_FILE_SPECS:
        search_path = REPO_ROOT / dir_name
        print(f"\nChecking '{dir_name}' (pattern '{pattern}') -> {search_path}")
        if not search_path.exists():
            print("  [missing] directory does not exist")
            continue
        all_files = sorted(p for p in search_path.rglob("*") if p.is_file())
        print(f"  {len(all_files)} file(s) present (before pattern matching):")
        #for p in all_files:
        #    print(f"    {p}")
        matched = sorted(search_path.rglob(pattern))
        print(f"  {len(matched)} match pattern '{pattern}'")
        files_to_upload.extend(matched)

    if not files_to_upload:
        print("No large result files found.")
        return

    total_bytes = sum(f.stat().st_size for f in files_to_upload)
    print(f"Found {len(files_to_upload)} files ({total_bytes / 1e9:.2f} GB)")

    if dry_run:
        for f in files_to_upload:
            print(f"  {f}")
        return

    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api = HfApi()

    # HF rate-limits *commits* (not data transfer), so making many small commits
    # is exactly what triggers 429s. upload_large_folder uploads files in
    # parallel, bundles them into a few commits, backs off on rate limits on its
    # own, and skips files already present unchanged. It is also RESUMABLE: if it
    # dies or you Ctrl-C, just run this script again and it continues where it
    # left off (progress is cached under <folder>/.cache/huggingface/).
    allow_patterns = sorted({f"{dir_name}/{pattern}" for dir_name, pattern in LARGE_FILE_SPECS})
    print(f"\nUploading via upload_large_folder with allow_patterns={allow_patterns}")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(REPO_ROOT),
        allow_patterns=allow_patterns,
        print_report=True,
    )

    print(f"\nDone. Pushed matching files to {repo_id}")


if __name__ == "__main__":
    fire.Fire(push)
