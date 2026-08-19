#!/usr/bin/env bash
# Regenerate the downstream-divergence artifacts with per-generation records:
# the interp boundary run, the decay run, and the temperature sweep. These are
# the runs behind WRITEUP.md's decay table and temperature figure; the JSONL
# records they write are published in the HF dataset so the tables are
# re-derivable on CPU (see the --from-jsonl mode of each module).
#
# Needs a GPU (Llama-3.2-1B and Qwen2.5-1.5B, bf16). On an A40:
#
#   sky launch skypilot/reproduce.yaml --infra runpod --gpus A40:1 -y \
#       --env HF_TOKEN=... \
#       --env RUN_CMD="bash scripts/regenerate_divergence_artifacts.sh"
set -euo pipefail
cd "$(dirname "$0")/.."

ART="${ART:-data/retok/artifacts}"
mkdir -p "$ART"

# Interp boundary run: 25 prompts x 24 samples at the seed and length recorded
# in RESULTS.md (600 generations).
uv run python -m retok.phase2_interp --model meta-llama/Llama-3.2-1B-Instruct \
    --n-samples 24 --max-new-tokens 200 --temperature 1.0 --seed 41 \
    --jsonl-out "$ART/interp_meta-llama_Llama-3.2-1B-Instruct.jsonl"

# Decay run. RESULTS.md never recorded the original invocation, so these are
# the module defaults, now pinned explicitly.
uv run python -m retok.phase2_decay --model meta-llama/Llama-3.2-1B-Instruct \
    --n-samples 8 --max-new-tokens 220 --temperature 1.0 --seed 0 \
    --jsonl-out "$ART/decay_meta-llama_Llama-3.2-1B-Instruct.jsonl"

# Temperature sweep: 48 generations per sampled cell (8 prompts x 6 samples);
# temperature 0 is greedy = deterministic, so 1 generation per prompt.
uv run python -m retok.phase2_temperature \
    --models meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
    --temperatures 0.0 0.7 1.0 1.5 2.0 \
    --n-samples 6 --max-new-tokens 160 --seed 0 \
    --jsonl-dir "$ART"

echo "Done. Records in $ART/."
