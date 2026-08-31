# retok_rl — will RL spontaneously learn to emit non-canonical tokens?

**Status:** two arms run — see [RESULTS.md](RESULTS.md) and WRITEUP.md §4.
Expansion arm (single seed, gpt2-large): canonical fell 100% → 61.5% of
rollouts over 2,000 GRPO steps under a reward blind to tokenization — mostly
reward-dense *greedy-longest* chunks (37.5%), with the single-digit
(compute-buying) form rising late and small (0 → 1.6%). Reversal arm (Arm C,
single seed, Qwen2.5-3B-Instruct, artifact-backed): the pre-registered
compute-attractor prediction was **not supported** — reward roughly doubled
while segmentation stayed put: per-token round-trip canonicality flat
(6.7% → 4.8%), all-single-char 0% of compliant answers at every eval,
single-char token share 20% → 10%. Pre-registered caveats apply (final
reward in the plan's weak-evidence band; the prefix-only reward put
selection pressure on ~2 characters). See the Run 5 metric-correction note
in RESULTS.md — segmentation is measured by round-trip canonicality, and
wrong-but-canonically-tokenized text counts as canonical.
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
