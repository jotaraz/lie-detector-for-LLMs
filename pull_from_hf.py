"""Download large result files from HuggingFace and place them in results/.

Files are downloaded to their original paths (results/experiment/file.json),
so they slot in alongside the small result files already tracked in git.

Usage:
    python pull_from_hf.py                         # download all large files
    python pull_from_hf.py --experiment foo__bar   # only one experiment folder
    python pull_from_hf.py --path autocomp2/data-autoconv3/  # any path substring
    python pull_from_hf.py --dry_run               # preview what would be downloaded
    python pull_from_hf.py --repo_id other/repo    # override repo
"""

from pathlib import Path

import fire
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "jo-chen/lie-detector-results"


def pull(
    repo_id: str = REPO_ID,
    experiment: str | None = None,
    path: str | None = None,
    dry_run: bool = False,
) -> None:
    api = HfApi()

    all_files = [
        f
        for f in api.list_repo_files(repo_id, repo_type="dataset")
        if not f.startswith(".")  # skip .gitattributes etc.
    ]

    if experiment is not None:
        all_files = [f for f in all_files if f"results/{experiment}/" in f]

    if path is not None:
        all_files = [f for f in all_files if path in f]

    if not all_files:
        filt = (
            f" for experiment '{experiment}'" if experiment else ""
        ) + (f" matching path '{path}'" if path else "")
        print(f"No files found{filt}.")
        return

    print(f"Found {len(all_files)} files in {repo_id}")

    if dry_run:
        for f in all_files:
            print(f"  {f}")
        return

    repo_root = Path(__file__).parent

    for i, file_path in enumerate(all_files, 1):
        local_path = repo_root / file_path
        if local_path.exists():
            print(f"[{i}/{len(all_files)}] Already exists, skipping: {file_path}")
            continue

        print(f"[{i}/{len(all_files)}] Downloading {file_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            local_dir=str(repo_root),
        )

    print(f"\nDone. Files are in {repo_root / 'results'}")


if __name__ == "__main__":
    fire.Fire(pull)
