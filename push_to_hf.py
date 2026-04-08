"""Push large result files to a HuggingFace dataset repo.

Large files (per-token score JSONs) are excluded from git and stored on HF instead.
Run this after an experiment to archive the large outputs.

Usage:
    python push_to_hf.py                          # push all large files
    python push_to_hf.py --dry_run                # preview what would be uploaded
    python push_to_hf.py --repo_id other/repo     # override repo
"""

from pathlib import Path

import fire
from huggingface_hub import HfApi, create_repo

# Set this to your HuggingFace repo (will be created if it doesn't exist)
REPO_ID = "jo-chen/lie-detector-results"

# These are the (dir, pattern) pairs excluded from git and stored on HF
LARGE_FILE_SPECS = [
    ("results", "*_all_tokens.json"),
    ("results", "control_scores.json"),
    ("autocomplete_results", "*.pt"),
]


def push(
    repo_id: str = REPO_ID,
    dry_run: bool = False,
) -> None:
    files_to_upload: list[Path] = []
    for dir_name, pattern in LARGE_FILE_SPECS:
        search_path = Path(dir_name)
        if search_path.exists():
            files_to_upload.extend(sorted(search_path.rglob(pattern)))

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

    for i, file_path in enumerate(files_to_upload, 1):
        size_mb = file_path.stat().st_size / 1e6
        print(f"[{i}/{len(files_to_upload)}] Uploading {file_path} ({size_mb:.0f} MB)...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=str(file_path),  # preserves results/experiment/file.json path
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"\nDone. Uploaded {len(files_to_upload)} files to {repo_id}")


if __name__ == "__main__":
    fire.Fire(push)
