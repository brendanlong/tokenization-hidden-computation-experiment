# Staged artifacts — pending upload to the HF dataset

These per-generation records are the 2026-08-19 re-measurement of the
downstream-divergence experiments (interp boundaries, contamination decay,
temperature sweep), produced by
[`scripts/regenerate_divergence_artifacts.sh`](../scripts/regenerate_divergence_artifacts.sh)
on a RunPod A40 (torch 2.11.0+cu128 via the driver fallback in
`skypilot/reproduce.yaml`, transformers 4.57.6). They back the decay/temperature
numbers in WRITEUP.md and the figures, and are re-derivable on CPU via each
module's `--from-jsonl` mode.

They are committed here **temporarily** because this branch was prepared with a
read-only HF token. To publish them to
[`brendanlong/retok-noncanonical-tokenization`](https://huggingface.co/datasets/brendanlong/retok-noncanonical-tokenization)
(which is where `hf:` paths, `scripts/reproduce_analyses.sh`, and the CI
drift-guard expect them):

```bash
HF_TOKEN=<write token> uv run python scripts/upload_artifacts.py \
    --local artifacts --prefix "" --include '*.jsonl'
```

Also worth doing in the same pass: the dataset card on HF is stale relative to
[`hf_dataset_card.md`](../hf_dataset_card.md), which now lists these files (and
the `greedy/` + `lw_comparison/` ones it was missing) — copy it over as the
dataset README.

After uploading, this directory can be removed from the repo — the dataset is
the canonical home, and everything in the repo reads it via `hf:` paths.
