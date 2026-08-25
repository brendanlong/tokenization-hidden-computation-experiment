# retok_rl — will RL spontaneously learn to emit non-canonical tokens?

**Status:** run — see [RESULTS.md](RESULTS.md). Headline (single seed,
gpt2-large): canonical rate fell 100% → 61.5% over 2,000 GRPO steps under a
reward blind to tokenization, drifting toward reward-dense *greedy-longest*
chunks — not the single-digit (compute-buying) form, which stayed ≈0.
See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the pre-registered predictions.

Follow-up to [`retok`](../retok/), which established that re-tokenized
transcripts can hide computation, that real models emit non-canonical tokens at
low rates, and that they can be *prompted* into it — but only argued
*structurally* that RL pressure would produce it.

This trains GPT-2 with GRPO on decimal expansion of `1/b`, rewarding the number
of correct leading digits **of the decoded text**, so any tokenization that
produces the right digits scores identically. The question is whether the policy
spontaneously moves to single-digit tokens, which a stored transcript would then
hide.

Prediction (pre-registered): the non-canonical rate goes to ~100% single-digit
tokens. Note there are **two** ways to drift off canonical — toward single
digits, or toward reward-dense long chunks — and greedy-longest-match differs
from BPE's segmentation for 93 of 98 divisors, so the null outcome requires
landing on precisely the canonical segmentation.
