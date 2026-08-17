#!/usr/bin/env bash
# Retrain the toy model and regenerate the wild-caught measurements.
#
# Needs a GPU. Core arms are enabled below; sweeps and controls are commented
# out with runtime notes — uncomment what you want.
#
# Runtimes are from an RTX 3060 (8 GB) for training and a RunPod A40 (44 GB)
# for generation; see RESULTS.md for the original per-run records.
#
# Set WANDB_MODE=disabled (or pass --no-wandb) to skip experiment tracking.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:-data/retok/checkpoints}"
ART="${ART:-data/retok/artifacts}"
mkdir -p "$CKPT" "$ART"

echo "=============================================================="
echo "Phase 1 — toy model, 3 seeds (~35 min each on an RTX 3060)"
echo "=============================================================="
# Defaults reproduce the headline: base 10, 3 digits, 2 layers, dim 16,
# direct_fraction 0.3, 20M streaming unique examples, batch 512, LR 3e-4
# cosine with 200 warmup steps.
for seed in 0 1 2; do
    uv run python -m retok.train \
        --seed "$seed" \
        --wandb-run-name "retok-main-s$seed" \
        --checkpoint-dir "$CKPT/retok-main-s$seed"
done

# --- Width sweep (appendix figure; ~4 h total) ------------------------------
# Locates the dim=16 regime where digit-by-digit succeeds and one pass fails.
# Depth cannot be the knob: depth=1 breaks CoT itself, since a CoT step must
# compute a digit *and* expose its carry.
# for d in 8 12 16 24 32 64; do
#     uv run python -m retok.train --dim "$d" --seed 0 --generate-n 15000000 \
#         --wandb-run-name "retok-sweep-cot-d$d" \
#         --checkpoint-dir "$CKPT/retok-sweep-cot-d$d"
#     uv run python -m retok.train --mode one_step --dim "$d" --seed 0 \
#         --generate-n 15000000 --wandb-run-name "retok-sweep-1step-d$d" \
#         --checkpoint-dir "$CKPT/retok-sweep-1step-d$d"
# done

echo
echo "=============================================================="
echo "Phase 2 — wild-caught generation (~20 min per model on an A40)"
echo "=============================================================="
# --dtype auto runs each model at its as-released precision, which matters:
# this is a tail-sampling measurement and dtype moves it (GPT-2 at bf16 rather
# than its native fp32 shifts Cyrillic 0.43% -> 1.09%).
#
# gpt-oss-20b needs ~44 GB VRAM; drop it if you have less. The Llamas are
# gated on HuggingFace and need `huggingface-cli login`.
MODELS=(
    gpt2
    meta-llama/Llama-3.2-1B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct
    google/gemma-2-2b
    meta-llama/Llama-3.2-3B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
    openai/gpt-oss-20b
)
# `|| echo` so one unavailable model (gated license, insufficient VRAM) doesn't
# abort the sweep before the rest have run.
for model in "${MODELS[@]}"; do
    uv run python -m retok.phase2_probe --model "$model" --dtype auto \
        --n-samples 12 --max-new-tokens 200 --temperature 1.0 --seed 31 \
        --jsonl-out "$ART/${model//\//_}.jsonl" \
        || echo ">>> SKIPPED $model (gated, OOM, or download failed)"
done

echo
echo "=============================================================="
echo "Phase 2 — downstream divergence (WRITEUP.md §1, decay table)"
echo "=============================================================="
# 25 prompts x 24 samples = the 600 generations / 149 non-canonical boundaries
# recorded in RESULTS.md, at its seed and length.
uv run python -m retok.phase2_interp --model meta-llama/Llama-3.2-1B-Instruct \
    --n-samples 24 --max-new-tokens 200 --temperature 1.0 --seed 41

# NOTE: RESULTS.md records the decay result (49 spans) but not the command that
# produced it, so these are the script defaults and will not land on exactly 49
# spans. The shape of the curve reproduces; the per-row n will differ.
uv run python -m retok.phase2_decay --model meta-llama/Llama-3.2-1B-Instruct

# --- Temperature sweep (~1 h) ----------------------------------------------
# 48 generations per cell (8 prompts x 6 samples), except temperature 0.0 which
# is deterministic and so runs 1 generation per prompt (8 per model).
# uv run python -m retok.phase2_temperature \
#     --models meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
#     --temperatures 0.0 0.7 1.0 1.5 2.0

# --- Prompted induction + matched controls (WRITEUP.md §3; ~20 min) --------
# for model in meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
#     uv run python -m retok.phase2_induce --model "$model" \
#         --jsonl-out "$ART/induce_${model//\//_}.jsonl"
# done

echo
echo "Done. Checkpoints in $CKPT/, generation records in $ART/."
echo "Verify the rates with: uv run python -m retok.phase2_verify $ART/*.jsonl"
