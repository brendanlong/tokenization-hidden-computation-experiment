"""Resolve published artifacts (checkpoints, generation records) from HuggingFace.

Everything this repo's analyses read is hosted in one public HuggingFace
dataset, `brendanlong/retok-noncanonical-tokenization`:

    <root>/                          per-model generation records (*.jsonl)
    checkpoints/<run_name>/final.pt  trained toy-model checkpoints
    checkpoints/<run_name>/final.pt.json   run metadata sidecar

`hf_hub_download` handles caching (under `~/.cache/huggingface` by default, or
`HF_HOME`), so repeated analysis runs hit the local cache. No account or token
is needed — the dataset is public.
"""

from __future__ import annotations

DATASET_REPO = "brendanlong/retok-noncanonical-tokenization"

# Checkpoints referenced by RESULTS.md. Mirrors the run names in the run log,
# so an `s3://.../retok/checkpoints/<run>/final.pt` URI there maps 1:1 onto
# `hf:checkpoints/<run>/final.pt` here.
CHECKPOINT_PREFIX = "checkpoints"

# Per-model generation records backing the wild-caught rate table. The prompted
# induction trials live in `induce_<model>.jsonl` and are analysed separately.
PUBLISHED_RECORD_FILES = (
    "gpt2.jsonl",
    "meta-llama_Llama-3.2-1B-Instruct.jsonl",
    "Qwen_Qwen2.5-1.5B-Instruct.jsonl",
    "google_gemma-2-2b.jsonl",
    "meta-llama_Llama-3.2-3B-Instruct.jsonl",
    "meta-llama_Llama-3.1-8B-Instruct.jsonl",
    "openai_gpt-oss-20b.jsonl",
)


def artifact_path(relpath: str, *, repo_id: str = DATASET_REPO) -> str:
    """Download `relpath` from the public dataset and return its local path.

    Args:
        relpath: Path within the dataset repo, e.g.
            ``checkpoints/retok-main-s0/final.pt`` or ``gpt2.jsonl``.
        repo_id: Override the dataset repo (for forks/mirrors).
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=relpath, repo_type="dataset")


def checkpoint_artifact(run_name: str, filename: str = "final.pt") -> str:
    """Local path to a published toy-model checkpoint, by run name."""
    return artifact_path(f"{CHECKPOINT_PREFIX}/{run_name}/{filename}")


def resolve_record_path(pathlike: str) -> str:
    """Resolve a records path that may point into the published dataset.

    ``hf:<relpath>`` downloads (or reuses the cache of) ``<relpath>`` from the
    public dataset and returns the local path; anything else is returned
    unchanged. Lets the ``--from-jsonl`` analysis modes run straight off the
    published artifacts with no manual download step.
    """
    if pathlike.startswith("hf:"):
        return artifact_path(pathlike[3:])
    return pathlike


def published_record_files() -> tuple[str, ...]:
    """Names of the per-model generation-record files in the dataset."""
    return PUBLISHED_RECORD_FILES
