# Results: retok_rl (will RL spontaneously learn non-canonical tokens?)

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for pre-registered predictions.
All runs: GRPO via TRL 1.9.2, reward = count of correct leading digits of the
**decoded** completion, task `1/b` to 30 places, 77 train / 21 held-out
divisors, temp 1.0 / top_k 0 / top_p 1.0 (TRL defaults), no SFT warm start.
Launched with `skypilot/train-retok-rl.yaml` on RunPod.

## Run 1 — gpt2, β=0, no entropy bonus (`retok-rl-beta0`) — INVALID

```
sky launch skypilot/train-retok-rl.yaml -c retok-rl-a --infra runpod --down --yes \
  --retry-until-up --env RUN_NAME=retok-rl-beta0 --env STEPS=3000 --env BETA=0.0 --env PLACES=30
```

A40 (CA), ~$0.90. **Entropy collapsed at step ~250**: entropy 4.65 → 0.004
nats, `frac_reward_zero_std` pinned at 1.0, so every GRPO group had identical
rewards and the gradient was zero. The remaining 2,750 steps were no-ops
(reward frozen at exactly 1.42, "canonical 100.0%" = one deterministic output,
not a preference). **Do not cite this run's tokenization numbers.** Led to the
`CollapseGuard` callback and the entropy default.

## Run 2 — gpt2, β=0, entropy_coef=0.05 (`retok-rl-ent05`)

```
sky launch ... --env RUN_NAME=retok-rl-ent05 --env STEPS=3000 --env BETA=0.0 \
  --env PLACES=30 --env ENTROPY_COEF=0.05
```

A40, ~$0.90. Exploration alive all 3,000 steps (greedy-longest flickering
0.5–2.4%, CollapseGuard never fired). Reward 0.97 → 1.44 of 30 and plateaued —
**gpt2 cannot do the task**, so its tokenization curve is weak evidence.
Single-digit 1.8% → 0.0%; canonical 97.6–99.5% throughout.

## Run 3 — meta-llama/Llama-3.2-1B-Instruct — FAILED (no access)

Two launches failed 401 on the gated repo. Root cause was on our side: the
verification that the token "worked" had read a **cached** config.json and
never hit the network; a fresh-cache fetch shows a clean `GatedRepoError`.
The token lacks access to Llama repos. (An intermediate `huggingface-cli login`
call was also a red herring — `whoami()` 401s for fine-grained tokens lacking
that scope even when model download works.)

## Run 4 — gpt2-large, β=0, entropy_coef=0.05 (`retok-rl-gpt2large`) — HEADLINE

```
sky launch skypilot/train-retok-rl.yaml -c retok-rl-large --infra runpod --down --yes \
  --retry-until-up --env RUN_NAME=retok-rl-gpt2large --env MODEL=openai-community/gpt2-large \
  --env STEPS=2000 --env BETA=0.0 --env PLACES=30 --env ENTROPY_COEF=0.05
```

A40, ~$0.60. gpt2-large chosen over Llama for the *identical tokenizer* to
gpt2 (channel held fixed, only capability changes) after the access failure.

| step | reward | single-digit | canonical | greedy-longest |
|---:|---:|---:|---:|---:|
| 50 | 1.35 | 0.0% | 100.0% | 0.0% |
| 400 | 2.06 | 0.0% | 99.0% | 0.5% |
| 800 | 2.47 | 0.0% | 95.2% | 3.4% |
| 1200 | 2.57 | 0.0% | 79.3% | 19.2% |
| 1600 | 2.67 | 0.3% | 69.2% | 27.9% |
| 2000 | 2.99 | 1.6% | **61.5%** | **37.5%** |

**Canonical 100% → 61.5% in 2,000 steps under a reward blind to tokenization**,
with the drift going to the *greedy-longest* attractor (reward-dense memorised
chunks — prediction 1's counter-case, called in advance in the plan), still
rising at the step cap. Single-digit stayed ≈0 (prediction 1's primary form did
not occur). gpt2 (run 2) at the same settings stayed ~99% canonical while
failing the task, so the drift appeared only with capability.

Caveats: single seed; reward 2.99/30, so the task is far from mastered; 77
memorised sequences, not division; held-out reward flat (~2.2) — no
generalisation.

> **Annotation (2026-08-30, recovered from wandb `viviclnk`; no per-rollout
> artifacts exist for this run).** Two clarifications to the table above:
> (a) the "single-digit" column is `frac_single_digit_tokens` — the fraction
> of emitted digit tokens of length 1 — NOT the all-single-digit *attractor*,
> which stayed 0.0–0.5% throughout and was 0.0% at step 2000; (b) mean digits
> emitted grew 2.1 → 3.0 → 3.1 → 4.5 → 5.0 → 5.2 across the tabulated steps,
> and longer digit runs mechanically depress the per-rollout canonical rate,
> so part of the canonical decline is a length effect. The "other" attractor
> stayed ≤2.4% at every eval — deviations from canonical landed almost
> exactly on greedy-longest rather than scattering — which is the evidence
> that the segmentation movement is real and specific rather than a length
> artifact alone. No per-token round-trip canonicality was computed for this
> arm and it cannot be recomputed retroactively (training logs only); a
> rerun with the Run-5-style artifact retention would close that gap.

## Infrastructure notes

RunPod image resolves Python 3.14 where `dill`/`datasets` breaks — pinned via
repo-level `.python-version` (3.12). Smoke branch must propagate the real exit
code (`PIPESTATUS`) or crashed runs report SUCCEEDED — this caught two later
failures. A missing `WANDB_API_KEY` no longer kills a run (falls back to
`--no-wandb`). Pass `token=` explicitly to `from_pretrained`; images may ship a
stored HF token that shadows the env var.

## Run 5 — Arm C: word reversal, Qwen2.5-3B-Instruct (`retok-rl-reversal-s0`)

```
sky launch skypilot/train-retok-rl.yaml -c retok-rl-rev --infra runpod --gpus A40:1 \
  --down --yes --retry-until-up --secret WANDB_API_KEY --env TASK=reversal \
  --env RUN_NAME=retok-rl-reversal-s0 --env STEPS=2000 --env MODEL=Qwen/Qwen2.5-3B-Instruct
```

A40, 1h00m training (~$0.45; plus ~$0.15 across three failed launches — see
infra notes). Pre-registered in EXPERIMENT_PLAN.md Arm C *before* launch.
Config: GRPO, batch 16, 16 generations/group, lr 1e-5, temp 1.0 pure
sampling, bf16, `entropy_coef=0.05` — the same value as the expansion arm,
chosen once after the no-bonus run (Run 1) collapsed, not tuned; `beta=0`;
`--stop-bonus` did not exist yet (reward was leading-correct only).
Artifacts: 16,564 per-rollout records (token IDs, targets, per-eval) in
`rollouts.jsonl`; step-0 baseline eval before any training. wandb:
`retok-rl-reversal-s0`, project `retok_rl`.

> **Metric correction (2026-08-30) — supersedes the tokenization numbers in
> earlier revisions of this entry and in PR #13's first description.** The
> original per-eval "canonical / greedy / other" attractor metric had two
> defects: it lowercased the produced surface before comparing segmentations
> (misfiling canonically-tokenized uppercase output as "other"), and it was
> computed over all rollouts, so text that was simply *wrong* — junk letters
> included — could move it. An unexpected-but-canonical token is not a
> non-canonical token: wrong text contributes zero to segmentation metrics.
> The tables below use the corrected metrics (now in `reversal.py`, all
> recomputable from `rollouts.jsonl`): **round-trip canonicality**
> (`encode(decode(ids)) != ids`, the project's core metric — per token, per
> answer-run, and per generation) for arbitrary emitted text, and the
> **attractor mix over compliant rollouts only** for segmentation of the
> answer. This also retracts the "transient rise of the compute attractor to
> 9.5% at step 150" claim from the previous revision — that spike was the
> defective metric counting short junk runs; under the corrected metric,
> all-single-char is 0% of compliant answers at every eval.

**What changed, step 0 → 2000** (train split; 41 evals in the artifacts):

| metric | 0 | 150 | 550 | 1150 | 2000 |
|---|---:|---:|---:|---:|---:|
| reward (leading correct chars; mean target len 5.5) | 0.92 | 1.04 | 1.61 | 1.86 | 1.94 |
| held-out reward | 1.08 | 0.84 | 1.40 | 1.67 | 1.61 |
| exact match | 6.0% | 0.6% | 1.2% | 1.8% | 0.6% |
| completion length ratio (produced/target) | 0.92 | 1.40 | 1.20 | 1.24 | 1.45 |
| single-char tokens over the letter run | 20.0% | 19.2% | 10.9% | 11.8% | 10.4% |
| **round-trip non-canonical, per token** | 6.7% | 3.9% | 5.9% | 4.5% | **4.8%** |
| **round-trip non-canonical, answer run only** | 6.7% | 8.3% | 8.3% | 4.2% | **6.0%** |
| **round-trip non-canonical, answer run, per token** | 6.1% | 8.4% | 8.4% | 4.4% | **5.2%** |
| round-trip non-canonical, per generation | 7.7% | 26.7% | 46.8% | 36.6% | 37.3% |
| excluded (U+FFFD, unmeasurable) | 0.0% | 1.8% | 17.3% | 20.2% | 10.7% |
| **single-char tokens, correct region only** | 43.4% | 55.3% | 11.1% | 1.9% | **0.0%** |
| **round-trip non-canonical, correct region** | 2.4% | 0.0% | 0.0% | 1.9% | **0.0%** |

Held-out tracks train on every metric (e.g. per-token round trip 4.2% →
4.7%). Notes on reading the table:

- The **per-generation** round-trip rate rose 7.7% → 37.3% while the
  **per-token** rate stayed flat — completions grew from ~2.5 tokens to the
  16-token cap, and a sequence-level flag accumulates with length
  (non-recovering BPE; the same length effect documented for Phase 2). The
  added tokens are unrewarded trailing text after the scored prefix: the
  reward stops counting at the first mismatch (and at target exhaustion), so
  output there is neither rewarded nor penalised.
- **Segmentation of the answer, where defined** (compliant rollouts, i.e.
  exact matches): canonical 91.2% pooled over steps 0–150 (n=34), 100%
  pooled over steps 1500–2000 (n=41); all-single-char 0% at every eval.
  Exact matches collapsed (6.0% → 0.6%; length-3 words 25% → 4%), so late n
  is small. Non-exclusive match rates (added 2026-08-30;
  `match_compliant/*` in `reversal.py`, same pooling): canonical 91.2%
  early / 100% late, greedy-longest 38.2% early / 100% late,
  all-single-char 0% throughout — every late compliant surface has
  canonical == greedy-longest (87.1% of compliant surfaces across all
  evals do), so the two references are indistinguishable there.
- The **correct-region rows** (added 2026-08-30; `correct_region/*` in
  `reversal.py`) are the primary segmentation metrics: computed per token,
  only over tokens lying entirely within the correct leading prefix of the
  answer, so wrong characters and junk are excluded by construction. The
  correct region is canonically tokenized at every eval (0–2.4%
  non-canonical) and its single-char share fell 43% → 0% — by the end the
  correct prefix (~2 chars) is carried by exactly one multi-char token
  (mean 1.26 → 1.00 tokens). Per-position: position 1 went 50% → 0%
  single-char-carried (n=42 → 63); positions 2–3 were multi-char-carried
  throughout; position 4+ was almost never reached.
- The **answer-run per-token row** (added 2026-08-30;
  `roundtrip/answer_tok_non_canonical` in `reversal.py`) round-trips the
  emitted letter run alone — `encode(decode(kept_ids))` vs `kept_ids`,
  diff tokens over run tokens — so it measures only the answer region and
  is not diluted or inflated by trailing text. It is flat: 6.1% → 5.2%
  train, 3.6% held-out at 2000, no trend across the 41 evals (range
  1.8–13.3%, the 11.8/13.3% values at steps 50/100 only).
- The step-0 per-token rate (6.7% train / 4.2% held-out) is far above
  Qwen2.5's Phase-2 english rate (0.00%); reversed-word text is
  off-distribution, the regime Phase 2 measured as high-rate.

**Prediction status.** P1 (single-char fraction rises with reward): not
supported — reward roughly doubled, single-char fell 20.0% → 10.4%, and
all-single-char is 0% of compliant answers throughout. P2 (held-out more
single-char than train at matched reward): no difference (late: 11.4% train
vs 12.5% held-out). P3 (capability gate): final reward 1.94/1.61 against
mean target length 5.5 sits inside the ~1–2 char band the plan marked in
advance as "weak evidence either way"; additionally, because the reward
scores only the leading correct prefix, selection pressure concentrated on
the first ~2 characters (full correct reversals fell 9.5% → 2.4% even
allowing trailing text). Single seed.

**Harness notes for future runs** (no further runs currently planned):
`--stop-bonus B` now exists (extra reward when the completion is exactly the
answer followed by EOS — trains the policy to stop after getting it right);
the prompt embeds the word in quotes, and the canonical encoding of `"word`
splits some words that would be single tokens after a space (`"garden` →
`['g','arden']`, vs ` garden` as one token) — canonical by construction, but
a future run wanting guaranteed single-token inputs should drop the quotes.

### Infra notes (attempts 1–3, all pre-training failures, ~$0.15 total)

1. uv.lock resolves torch 2.13 (cu130 wheel); RunPod fleet is driver 570 /
   CUDA 12.8 → `cuda_available=False`, GPU preflight aborted. Fix: swap in
   torch 2.11.0+cu128 after `uv sync`.
2. uv's default first-match index strategy resolved torch from PyPI (no
   +cu128 builds) → `--index-strategy unsafe-best-match`.
3. Plain `uv run` re-synced the venv to the lockfile at job start, reverting
   the swap into a mixed cu128/cu13 env that died on import
   (`libtorch_cuda.so: undefined symbol: ncclCommResume`). Fix: `uv run
   --no-sync` everywhere + purge the lock's cu13 CUDA stack before the swap.
   The rollouts also now travel gzip+base64 in the log stream as an
   artifact escape hatch (an unresolvable ssh alias ate the Run-4-era
   fetch path twice). Extracting the blob from a `sky logs` capture
   requires stripping ANSI colour codes *before* removing the
   `(cluster, pid=N)` prefixes — `scripts/extract_rollouts_from_log.sh`
   does both and round-trips byte-identically.

## Run 6 — gpt2-large expansion replication with artifacts (`retok-rl-gpt2large-r2`)

```
sky launch skypilot/train-retok-rl.yaml -c retok-rl-exp --infra runpod --gpus A40:1 \
  --down --yes --retry-until-up --secret WANDB_API_KEY --env TASK=expansion \
  --env RUN_NAME=retok-rl-gpt2large-r2 --env STEPS=2000 --env MODEL=openai-community/gpt2-large
```

A40, 1h07m (~$0.50). Identical config to Run 4 (2,000 steps, β=0, 30 places,
`entropy_coef=0.05`, seed 0); purpose pre-noted in EXPERIMENT_PLAN.md —
artifact collection, since Run 4 retained no per-rollout records. This run
writes them (15,416 records incl. a step-0 baseline Run 4 lacked; wandb
`retok-rl-gpt2large-r2`) and computes the corrected metric families live.
All numbers below recompute from `rollouts.jsonl` via `metrics.summarise`.

**What changed, step 0 → 2000** (train split):

| metric | 0 | 400 | 800 | 1200 | 1600 | 2000 |
|---|---:|---:|---:|---:|---:|---:|
| reward (correct leading digits of 30) | 0.27 | 1.57 | 2.28 | 2.48 | 2.74 | 2.72 |
| digits emitted (mean) | 3.0 | 3.1 | 3.1 | 3.2 | 4.6 | 4.9 |
| digit-run tokens (mean) | 1.43 | 1.02 | 1.03 | 1.08 | 1.56 | 1.66 |
| canonical attractor (emitted digit run, right or wrong) | 89.9% | 98.1% | 98.6% | 96.2% | 76.0% | **71.2%** |
| greedy-longest attractor | 0.5% | 1.9% | 1.4% | 3.8% | 23.6% | **27.9%** |
| all-single-digit attractor | 6.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| other | 2.4% | 0.0% | 0.0% | 0.0% | 0.5% | 1.0% |
| matches canonical (non-exclusive) | 97.1% | 98.1% | 98.6% | 96.2% | 76.0% | 71.2% |
| matches greedy-longest (non-exclusive) | 76.8% | 100.0% | 99.5% | 100.0% | 99.5% | **98.1%** |
| matches all-single-digit (non-exclusive) | 6.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| single-digit tokens (share of digit tokens) | 13.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **round-trip non-canonical, per token (full completion)** | 0.55% | 8.24% | 4.95% | 5.52% | 6.99% | **6.11%** |
| **round-trip non-canonical, digit run, per token** | 4.1% | 3.8% | 2.8% | 7.1% | 31.1% | **35.3%** |
| round-trip non-canonical, digit run, per generation | 2.9% | 1.9% | 1.4% | 3.8% | 24.0% | 28.8% |
| correct-region single-digit | 83.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| correct-region non-canonical | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.7% |
| excluded (U+FFFD) | 0.5% | 16.8% | 23.1% | 20.2% | 23.1% | 27.4% |

Held-out at 2000: reward 2.28, canonical 68.5%, greedy-longest 29.8%,
per-token round trip 6.46% (digit run alone: 38.3%) — tracks train on
every metric.

Reading notes:

- The **non-exclusive match rows** (added 2026-08-30; `match/*` in
  `metrics.py`; denominator = rollouts with a non-empty digit run) compare
  the emitted run against each reference segmentation independently,
  because the references frequently coincide: canonical == greedy-longest
  for ~98% of emitted surfaces mid-training (runs of 3.0–3.2 digits) and
  69% at step 2000. The exclusive attractor rows above force a precedence
  on ties (all-single-digit, then canonical, then greedy-longest), which
  hides the overlap — e.g. at step 0 the exclusive canonical row reads
  89.9% but 97.1% of runs match the canonical segmentation (the 6.7%
  filed under all-single-digit were single-digit *and* canonical). Read
  side by side: **greedy-longest match reaches ~100% by step 400 and stays
  98–100% for the entire run**; canonical match equals it while the
  references coincide and falls 97.1% → 71.2% as reward lengthens the
  digit runs (3.0 → 4.9) and the references diverge. Restricted to the
  surfaces where the references differ, the emitted run matched
  greedy-longest 94–100% of the time from step 400 on (n=4–62 per eval),
  vs 97.7% canonical untrained (n=44).
- The **digit-run rows** (added 2026-08-30;
  `roundtrip/digitrun_tok_non_canonical` and `mean_digitrun_tokens` in
  `metrics.py`; the reversal analogues are `answer_tok_non_canonical` and
  `mean_answer_tokens`) round-trip the
  emitted digit run alone — `encode(decode(kept_ids))` vs `kept_ids`, diff
  tokens over run tokens — so the answer region is measured by itself
  rather than diluted across the ~38-token completion (at step 2000 the
  digit run is ~2 of those tokens, which is why the full-completion rate
  sits at 6.1% while a third of the answer tokens are non-canonical). The
  concentrate-then-chunk arc is directly visible in this row: 4.1%
  untrained → below 1% at several early evals (0.88% at step 50, 0.91% at
  500, 0.95% at 750 in the per-50-step artifact series) → 35.3% at the
  cap. The per-generation digit-run row lands within ~1pt of
  greedy-longest + other at every late eval, as it must — a digit run is
  non-canonical iff it deviates from the canonical segmentation.
- **Length-conditioned control** (added 2026-08-30, post-hoc): answers
  that are a single token are 0.0% round-trip non-canonical at every eval
  (a lone digit token is always its own canonical encoding here), so the
  aggregate per-token rise is substantially composition — 2+-token answers
  grew from 3% of rollouts at step 800 to 62% at step 2000. Conditioning
  on 2+-token answers, the non-canonical rate still rises 9.4% (step 0)
  → 46.9% (step 2000), but that residual tracks the tokenizer's own
  geometry for the longer strings being emitted: canonical coincides with
  greedy-longest for 96% of 3-digit strings but only 48% of 5-digit
  strings (17% at 4 digits, 8% at 12 — non-monotone), so an always-greedy
  policy on the step-2000 strings is ~52% non-canonical mechanically.
  Token-count comparison for the 128 multi-token step-2000 answers:
  canonical segmentation of the same digits uses the same number of
  tokens in 87 and MORE tokens in 41, never fewer — the emitted form is
  never longer than canonical and is strictly more digits-per-token in a
  third of cases. Net reading: reward lengthened answers and the policy
  extends locked-in tokens; the non-canonical rate at the seams is then a
  deterministic property of gpt2's digit tokenization at that length, not
  something the reward selected for directly (it cannot see it).
- **Lock-in + extension check** (added 2026-08-30, post-hoc): of the 26
  train divisors in the step-2000 eval, 13 modal answers are still a
  single token; of the 13 that grew to two tokens, 11 have a first token
  identical to that divisor's modal single-token answer from steps
  800–1200 (e.g. b=61 `016` → `016|993`), and the 2 exceptions changed
  digits, not segmentation (b=53 `016` → `018|604`, correcting toward
  0.01886…). All 13 first tokens are canonical for their own surface in
  isolation. Consistent with late answers being built by appending a
  second 3-digit chunk to a locked-in first token, with non-canonicality
  arising at the seam — the canonical segmentation of the joined string
  re-chunks across the boundary (`011`+`33` re-encodes as `01`+`133`).
  This also accounts for the greedy-longest direction: the locked-in
  first token is almost always the longest token prefix of the extended
  answer, i.e. greedy's first step. Caveats: modal answers, n=13, single
  seed, post-hoc.
- **Replicates Run 4's direction at somewhat smaller magnitude**: canonical
  fell to 71.2% here vs 61.5% in Run 4, greedy-longest reached 27.9% vs
  37.5%, reward 2.72 vs 2.99, digits emitted 4.9 vs 5.2. Same config, new
  GPU/nondeterminism; treat the two as the run-to-run spread.
- **The step-0 baseline (which Run 4 lacked) changes the shape of the
  story**: untrained gpt2-large is only ~90% canonical on its emitted digit
  runs, with 14% single-digit tokens and a 6.7% all-single-digit attractor.
  Early RL first *concentrates onto canonical* (98.6% by step 800, single-
  digit tokens → 0 immediately) and only later drifts to greedy-longest. So
  "canonical 100% → 61.5%" (Run 4's step-50 start) understates the untrained
  baseline's non-canonicality and overstates the peak: the RL arc is
  concentrate-then-chunk.
- **Per-token round-trip canonicality rose 0.55% → ~6%** — genuine
  segmentation drift under the same metric that stayed flat for the Qwen
  reversal arm (6.7% → 4.8%). This is the clean cross-arm contrast.
- Correct-region metrics are less informative for this arm: correct leading
  digits are few (~2–5), short digit strings' canonical and greedy
  segmentations usually coincide, and a long memorised token straddling the
  correct/incorrect boundary is excluded from the region by construction.
  Report them, but the digit-run attractor and the per-token round trip
  carry the story here.
- The U+FFFD exclusion rate grows to ~27% (trailing bytes after the digit
  run); measurable-subset caveat as in Run 5.
