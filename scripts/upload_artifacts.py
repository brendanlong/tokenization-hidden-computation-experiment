"""Maintainer script: publish checkpoints to the HuggingFace dataset.

The generation records (`*.jsonl`) are produced by `retok.phase2_probe
--jsonl-out` and uploaded the same way. This mirrors a local directory into the
dataset repo, so it is safe to re-run after adding a run.

    HF_TOKEN=hf_xxx uv run python scripts/upload_artifacts.py \
        --local data/retok/checkpoints --prefix checkpoints

Needs a token with *write* access to the target namespace; a read-only token
fails with "You don't have the rights to create a dataset under the namespace".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

from common.artifacts import DATASET_REPO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local", required=True, help="Local directory to mirror into the dataset"
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Path prefix inside the dataset repo (e.g. 'checkpoints')",
    )
    parser.add_argument("--repo", default=DATASET_REPO)
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would be uploaded and stop"
    )
    args = parser.parse_args()

    local = Path(args.local)
    if not local.is_dir():
        print(f"Not a directory: {local}", file=sys.stderr)
        return 1

    files = sorted(p for p in local.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f">>> {len(files)} files, {total / 1e6:.1f} MB -> {args.repo}/{args.prefix}")
    for p in files:
        print(f"    {p.relative_to(local)}")
    if args.dry_run:
        return 0

    api = HfApi()
    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    print(f">>> authenticated as {who.get('name')} (token role: {role})")
    if role == "read":
        print(
            ">>> ERROR: read-only token. Create a WRITE token at "
            "https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(local),
        path_in_repo=args.prefix,
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"upload {args.prefix or 'artifacts'}",
    )
    print(f">>> published: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
