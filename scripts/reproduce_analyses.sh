#!/usr/bin/env bash
# Reproduce every table in WRITEUP.md from published artifacts.
#
# No GPU, no wandb account. Downloads ~23 MB of checkpoints and ~6 MB of
# generation records from the HuggingFace dataset (cached by huggingface_hub,
# so re-runs are offline).
#
# The Llama and Gemma tokenizers are gated on HuggingFace: verifying those
# models' rates needs an accepted license plus `huggingface-cli login`. They are
# skipped with a clear message otherwise; everything else runs with no account.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-data/retok/reproduce}"
mkdir -p "$OUT"

echo "=============================================================="
echo "Phase 1 — the controlled toy (3 seeds)"
echo "=============================================================="
# Accuracy split (CoT vs single merged token), position collapse 3 -> 1,
# carry probe on the real stream vs the re-tokenized replay, and the
# calibration detector. WRITEUP.md §1.
for seed in 0 1 2; do
    echo "--- retok-main-s$seed ---"
    uv run python -m retok.analysis \
        --checkpoint "hf:checkpoints/retok-main-s$seed/final.pt" \
        --out "$OUT/analysis-s$seed"
done

echo
echo "=============================================================="
echo "Phase 2 — wild-caught rates, recomputed from raw token IDs"
echo "=============================================================="
# Deliberately recomputes canonicality from the emitted IDs rather than
# trusting the stored flags, so this audits the analysis and not just the
# bookkeeping. WRITEUP.md §2 rate table.
#
# Drop --all-published for one model, e.g.:
#   uv run python -m retok.phase2_verify hf:gpt2.jsonl
uv run python -m retok.phase2_verify --all-published

echo
echo "=============================================================="
echo "Figures"
echo "=============================================================="
# Numbers are the official runs recorded in RESULTS.md, so this regenerates
# figures/*.png byte-for-byte from the committed values.
uv run python -m retok.figures

echo
echo "Done. Analysis JSON in $OUT/, figures in figures/."
