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
