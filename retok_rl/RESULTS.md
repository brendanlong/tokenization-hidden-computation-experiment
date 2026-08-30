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

## Infrastructure notes

RunPod image resolves Python 3.14 where `dill`/`datasets` breaks — pinned via
repo-level `.python-version` (3.12). Smoke branch must propagate the real exit
code (`PIPESTATUS`) or crashed runs report SUCCEEDED — this caught two later
failures. A missing `WANDB_API_KEY` no longer kills a run (falls back to
`--no-wandb`). Pass `token=` explicitly to `from_pretrained`; images may ship a
stored HF token that shadows the env var.

## Run 5 — Arm C: word reversal, Qwen2.5-3B-Instruct (`retok-rl-reversal-s0`) — NULL for the compute attractor, with a reward hack

```
sky launch skypilot/train-retok-rl.yaml -c retok-rl-rev --infra runpod --gpus A40:1 \
  --down --yes --retry-until-up --secret WANDB_API_KEY --env TASK=reversal \
  --env RUN_NAME=retok-rl-reversal-s0 --env STEPS=2000 --env MODEL=Qwen/Qwen2.5-3B-Instruct
```

A40, 1h00m training (~$0.45; plus ~$0.15 across three failed launches — see
infra notes). Pre-registered in EXPERIMENT_PLAN.md Arm C *before* launch.
Per-rollout artifacts retained this time: 16,564 records (token IDs, targets,
attractor, correctness per eval) in `rollouts.jsonl`; step-0 baseline eval
runs before any training. wandb: `retok-rl-reversal-s0`, project `retok_rl`.

| step | reward | exact | attempted | single-char toks | canonical | greedy-longest | other | held reward |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.92 | 6.0% | 95.2% | 20.0% | 85.7% | 1.2% | 9.5% | 1.08 |
| 550 | 1.61 | 1.2% | 85.7% | 10.9% | 79.2% | 4.2% | 15.5% | 1.40 |
| 1150 | 1.86 | 1.8% | 91.7% | 11.8% | 81.5% | 0.6% | 17.9% | 1.67 |
| 2000 | **1.94** | **0.6%** | 89.3% | **10.4%** | **73.2%** | 3.6% | **22.6%** | **1.61** |

**Prediction 1 (primary) is not supported.** Reward roughly doubled
(0.92 → 1.94 leading chars; held-out 1.08 → 1.61) while the fraction of
single-character tokens *fell* from 20% to ~10% (19.9% → 7.0% restricted to
the tokens actually covering the target region) — on the task built to favour
the compute attractor, with the channel measurably open (canonical
segmentations of reversed words are 81% multi-char tokens in this tokenizer).
The all-single-char attractor rose only transiently: 1.8% at step 0 to a peak
of **9.5%** of train rollouts at step 150 (held-out 9.3%), decaying below
baseline as the prefix hack took over. Greedy-longest stayed in baseline
noise (0.0–6.5%).

**The canonical decline (85.7% → 73.2%) is non-compliance, not segmentation
drift.** Nothing in the reward penalises trailing garbage or casing. The
policy learned to emit a correct 2–3 character prefix followed by junk
(len-ratio rose 0.92 → 1.45; exact-match collapsed even as leading-correct
rose, len-3 words 25% → 4%; the grown "other" bucket is prefix-plus-junk:
`car → ['rac','HttpPost']`, `nut → [' tut','ген','Pizza']`), and uppercase
output rose from 0.7% to 18.3% of letter runs — which the classifier counts
as "other" by construction, since it compares segmentations case-sensitively
against the lowercase target. Canonical falls about as much among
length-matched rollouts (91.5% → 72.7%), so the decline tracks these
compliance-surface changes, not a re-segmentation of correctly produced
strings.

**Prediction 3 (capability gate) is partly triggered — stated up front, so
stating it now.** The plan designated reward staying at ~1–2 chars as "a
capability null... weak evidence either way". Final reward (1.94 train /
1.61 held-out, against mean target length 5.55) sits at the top of that
band, and the reward hack means the serial-reversal mechanism was under
selection pressure only for the first ~2 characters (full correct reversals,
even allowing trailing junk, fell 9.5% → 2.4%). This arm is therefore a
null with caveats, not a clean refutation: the defensible claim is "under
this reward, on the most favourable task we could build, the compute
attractor did not emerge in 2,000 steps".

**Prediction 2 (held-out more single-char than train at matched reward): no
support.** Late-training single-char fractions are 11.4% (train) vs 12.5%
(held-out) — flat.

**Reading.** On a modern instruct model with a modern tokenizer, 2,000 GRPO
steps of tokenization-blind reward on the compute-aligned task produced no
sustained movement into either non-canonical attractor — the policy moved
toward chunkier tokens and non-compliant surface forms (junk suffixes,
uppercase), not toward alternative segmentations of the target. The
contrast with Run 4 (gpt2-large drifting to greedy-longest chunks) says drift
is task- and lineage-dependent, not a general RLVR property. Design lesson
for any follow-up arm: leading-correct reward without an exactness/termination
term invites the prefix+junk hack; use exact-match bonus or penalise trailing
text.

Caveats: single seed; reward far from mastery (1.94 of mean target length
~5.5); 2,000 steps; the reward hack contaminates the canonical-attractor
metric, and the single-char token fraction is itself diluted by junk
(non-ASCII letters appear in 10.1% → 39.4% of letter runs over training,
and each multi-char junk token inflates the denominator) — restricted to
tokens covering the target region it falls 19.9% → 7.0%, same direction,
stronger.

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
