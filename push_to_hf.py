"""Push large result files to a HuggingFace dataset repo.

Large files (per-token score JSONs) are excluded from git and stored on HF instead.
Run this after an experiment to archive the large outputs.

Usage:
    python push_to_hf.py                          # push new/changed files only
    python push_to_hf.py --dry_run                # preview what would be uploaded
    python push_to_hf.py --repo_id other/repo     # override repo
"""

import hashlib
import time
from pathlib import Path

import fire
from huggingface_hub import CommitOperationAdd, HfApi, create_repo

# Set this to your HuggingFace repo (will be created if it doesn't exist)
REPO_ID = "jo-chen/lie-detector-results"

# These are the (dir, pattern) pairs excluded from git and stored on HF
LARGE_FILE_SPECS = [
    ("results", "*_all_tokens.json"),
    ("results", "control_scores.json"),
    ("autocomplete_results", "*.pt"),
    ("autocomp2/data-autoconv3", "*.pt"),
    ("autocomp2/data-autoconv4", "*.pt"),
    ("autocomp2/data-autoconv5", "*.pt"),
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

    # Build a map of {path_in_repo: sha256} for files already on HF
    print("Fetching existing file list from HuggingFace...")
    remote_sha256: dict[str, str] = {}
    try:
        for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
            if item.lfs is not None:
                remote_sha256[item.path] = item.lfs.sha256
    except Exception:
        pass  # repo may be empty/new — treat everything as new

    def local_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    to_upload = []
    skipped = 0
    for file_path in files_to_upload:
        repo_path = str(file_path)
        if repo_path in remote_sha256 and local_sha256(file_path) == remote_sha256[repo_path]:
            skipped += 1
        else:
            to_upload.append(file_path)

    print(f"  {skipped} file(s) unchanged and skipped, {len(to_upload)} to upload.")

    batch_size = 20
    batches = [to_upload[i:i + batch_size] for i in range(0, len(to_upload), batch_size)]
    for b_idx, batch in enumerate(batches, 1):
        total_mb = sum(f.stat().st_size for f in batch) / 1e6
        paths = ", ".join(str(f.name) for f in batch[:3])
        if len(batch) > 3:
            paths += f" (+{len(batch) - 3} more)"
        print(f"[batch {b_idx}/{len(batches)}] {len(batch)} files ({total_mb:.0f} MB): {paths}")
        operations = [
            CommitOperationAdd(path_in_repo=str(f), path_or_fileobj=str(f))
            for f in batch
        ]
        for attempt in range(5):
            try:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Upload batch {b_idx}/{len(batches)}",
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    print(f"\nDone. Uploaded {len(to_upload)} file(s) to {repo_id}")


if __name__ == "__main__":
    fire.Fire(push)
