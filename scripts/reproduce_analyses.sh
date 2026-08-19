#!/usr/bin/env bash
# Reproduce every table in WRITEUP.md from published artifacts.
#
# No GPU, no wandb account. Downloads ~0.3 MB of checkpoints and ~6 MB of
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
echo "Phase 1 — width sweep, re-evaluated from published checkpoints"
echo "=============================================================="
# All three columns of the appendix width-sweep table (CoT, one-step per-digit,
# one-step merged token). The dim=16 mixture arm is the headline model itself
# (retok-main-s0); there is no retok-sweep-cot-d16.
uv run python -m retok.eval_sweep

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
echo "Phase 2 — rates by the script the tokens are actually in"
echo "=============================================================="
# The domain labels are the PROMPT's language, not the output's: the small
# Llamas answer Russian and Japanese prompts in English, so their multilingual
# cells measure mostly-Latin text. This re-attributes each token to its own
# script, which is comparable across models. WRITEUP.md §2.
uv run python -m retok.phase2_script --all-published

echo
echo "=============================================================="
echo "Phase 2 — downstream divergence, re-derived from published records"
echo "=============================================================="
# The interp boundary report, the decay table, and the temperature sweep, from
# per-generation records. Canonicality/span bookkeeping is recomputed from the
# raw token IDs; the KL and probability values are read from the records (only
# regenerating them needs a GPU — scripts/regenerate_divergence_artifacts.sh).
# The Llama files need an accepted license for the (gated) tokenizer and are
# skipped with a message otherwise; the Qwen temperature file needs no account.
uv run python -m retok.phase2_interp \
    --from-jsonl hf:interp_meta-llama_Llama-3.2-1B-Instruct.jsonl
uv run python -m retok.phase2_decay \
    --from-jsonl hf:decay_meta-llama_Llama-3.2-1B-Instruct.jsonl
uv run python -m retok.phase2_temperature --from-jsonl \
    hf:temperature_meta-llama_Llama-3.2-1B-Instruct.jsonl \
    hf:temperature_Qwen_Qwen2.5-1.5B-Instruct.jsonl

echo
echo "=============================================================="
echo "Figures"
echo "=============================================================="
# Numbers are the official runs recorded in RESULTS.md, so this regenerates
# figures/*.png byte-for-byte from the committed values.
uv run python -m retok.figures

echo
echo "Done. Analysis JSON in $OUT/, figures in figures/."
