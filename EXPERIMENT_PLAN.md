# Experiment Plan: Hidden Computational Composition via Retokenization

**Status:** Phase 1 — design locked, running the official version.

## Hypothesis

A model's *actual* token stream can differ from the canonical re-tokenization
of its output in a way that makes per-position interpretability and monitoring
analysis wrong. BPE decoding is many-to-one: a multi-token derivation whose
concatenation is a single vocabulary item collapses to one token when a
transcript is stored as text and re-encoded. The stored transcript then shows
the model producing *in one token* what it actually produced over *several
decode steps* — so any analysis that operates on positions (logit lens, linear
probes, per-position attribution, serial-depth / compute accounting) is
attributing computation to positions that never ran.

The claim is about **steps/positions, not information or complexity**:
- We do **not** claim the computation is complexity-theoretically impossible in
  one pass (addition is TC⁰; a large enough model one-passes it). We claim that
  **this specific model** cannot produce the answer in the single-token form —
  it needs the multi-token derivation — and the transcript hides that.
- We do **not** claim information is destroyed. The answer's carries stay
  recoverable from the collapsed transcript (they are a function of the
  operands, which attention sees at every position). What is destroyed is the
  **positional structure** of the computation.

## Task: reversed addition, uncued CoT / merged-token mixture

A small decoder-only transformer is trained from scratch on 3-digit base-10
reversed addition (LSB-first, so carries propagate left-to-right). Operands are
given as **individual digit tokens** (`<bos> a0 a1 a2 + b0 b1 b2 =`) so the
model can read the digits it must add. The **merged vocabulary** has one token
per 3-digit answer string (1000 tokens) and is used only for the answer.

The answer is emitted in one of two formats, **mixed without a cue** in the
prompt:
- **CoT** (majority): digit-by-digit, `d0 d1 d2 <eos>` — three answer tokens,
  three decode steps, each with its own forward pass.
- **direct** (`direct_fraction`): a single merged token `[d0d1d2] <eos>` — one
  decode step.

Because the mixture is uncued, the model's distribution at `=` genuinely splits
between "emit a digit" and "emit the merged token", so re-tokenizing a CoT
answer (canonical greedy longest-match collapses the three digit tokens into the
merged token) yields a byte-identical, bona-fide sample of the direct format —
exactly this task's direct encoding of the same `(a,b)`. "Canonical replay" and
"direct generation" are the same token stream.

Sums stay within 3 digits (no overflow) so every answer is exactly one merged
token.

## The model (locked): dim=16, 2 layers

The width is chosen deliberately (see the sweep, Appendix): **dim=16** is the
regime where CoT succeeds (100%) but the single-token direct form genuinely
fails (~1–2%, one-step P(correct) ≈ 0.002). At larger width the model *can*
produce the answer in one pass and only fails to *tokenize* it — there the
transcript would not be hiding a real multi-step computation, so the claim would
be false. dim=16 is the honest regime.

## Predictions (pre-registered)

1. **This model needs the extra steps.** CoT accuracy ≈ 100% across carry-chain
   lengths; the single-token direct accuracy ≈ 0, with one-step P(correct) ≈ 0.
   → *Refuted if* the direct form also succeeds (the model doesn't need the
   multi-token derivation).

2. **Positions collapse; per-position accounting is wrong.** The real CoT answer
   occupies 3 decode positions; the canonical replay occupies 1. An analyst
   doing serial-depth / per-position accounting on the stored transcript
   undercounts the decode steps 3→1.

3. **Calibration detector (text + logprobs only).** Re-tokenized correct CoT
   transcripts are 100% correct, yet the model assigns each merged answer only
   ~0.002 one-step probability. Over a dataset this is astronomically unlikely
   as one-pass generation — a check that the stored transcript is not what ran,
   needing no token-ID logging.

4. **Per-position interpretability misleads on the replay.** Logit lens /
   attention at the single merged position reflect the model's (rarely-trained,
   ~OOD) handling of a re-encoded token, not the digit-by-digit trajectory that
   actually produced the answer. The per-position "answer forming" trajectory
   visible on the real CoT stream is absent on the replay.
   → Honest caveat: a probe for the *carry* recovers it on both streams (it is a
   function of the operands), so we frame (4) as *positional-trajectory* /
   attribution breakdown, not information loss.

## Analyses

- Per-format, per-carry-chain-length accuracy + one-step P(correct) (Prediction
  1, 3).
- Decode-position count, real vs replay (Prediction 2).
- Carry probe at each CoT computing-position vs the collapsed replay position —
  documents that information survives but positions don't (Prediction 4 caveat).
- Logit-lens / attention trajectory on the real CoT stream vs the replay
  (Prediction 4).

## Appendix experiments (supporting, not headline)

- **Width sweep (dim 8–64), 2 layers:** CoT stays 100% down to dim=16 while the
  one-pass forms (single merged token *and* a retrieval-free parallel per-digit
  readout) collapse — locating the dim=16 regime and showing CoT tolerates far
  more width-compression than one-pass. This is *how we picked the model*.
- **Why width, not depth:** depth=1 breaks CoT itself (a CoT step must compute a
  digit *and* expose its outgoing carry, wanting ~2 layers; at D=3, L=1 CoT ≈
  9%). So 2 layers is the CoT floor, and width is the only free knob that breaks
  the one-pass form while keeping CoT.
- **Retrieval vs computation (matched control):** a same-architecture model
  trained purely to one-step via parallel per-digit heads plateaus at ~10% at
  dim=16 (a capacity floor, not undertraining), and at dim=64 hits 100% while
  the merged-token form still fails — isolating retrieval as a separate effect.
  Used only to justify the dim choice; not needed for the core claim.

## Phase 2 (done — see RESULTS.md)

> **Stale numbers (annotation).** The figures in this paragraph are from the
> early, confounded run (`top_p=0.95`, per-model inherited sampling params,
> Llama's decode cleanup on) and were superseded by the controlled
> re-measurement — including the "Qwen2.5 ≈0%" claim, which was retracted.
> RESULTS.md and WRITEUP.md carry the correct tables; this paragraph is kept
> as the pre-registration-era snapshot.

Wild-caught result: open models sampled while retaining token IDs, find
naturally non-canonical spans (`encode(decode(gen_ids)) != gen_ids`), compare
the actual vs re-tokenized stream. Outcome: real models generate non-canonically
at 6–33% (Llama-3.2-1B, GPT-2; digit-individual Qwen2.5 ≈0%), and re-tokenizing
flips the top-1 next token for 23% of non-canonical boundaries (mean KL 0.51
nats). Nuance vs the plan: the wild phenomenon is mostly **subword** splitting,
not the toy's **digit**-merging (instruct models emit numbers canonically; GPT-2
does show the digit-mirror). So Phase 2 is the existence proof; the toy carries
the controlled hidden-computation claim.

## Explicitly out of scope

Complexity-theoretic (model-independent) one-pass impossibility (needs an
NC¹-hard task like S5, which has worse training + collapse-vocab problems);
binary / large-D variants (the retrieval wall or over-parallelizability made
them worse than base-10 D=3).

## Merged-operands variant (pre-registered 2026-08-30, before any training)

A rerun of the headline model with two changes: `direct_fraction=0.5` (was
0.3) and `--merged-operands` — DIRECT-format training examples encode
operands as merged tokens (`<bos> [750] + [860] = [521]`), so the fully
re-tokenized CoT transcript is now token-for-token an in-distribution
training format (pinned by `test_merged_operands_direct_equals_canonicalized_cot`)
instead of a sequence the model has never read. CoT examples are unchanged.
`generate_n=30M` (was 20M) so the digit branch keeps ~15M examples at the
smaller mixture share (the width sweep showed dim-16 CoT undertrained below
~10M). 3 seeds, everything else identical to `retok-main-s*`.

**Purpose.** Closes the §1 limitation that the re-tokenized replay is
off-distribution: with this model, "can't read this input" is eliminated as
an explanation, isolating "can't compute it in one step" and the
positional-structure probe result.

**Predictions.**
1. CoT (digit-by-digit) accuracy stays ≳99% at 30M examples.
2. Direct/merged accuracy stays far below CoT. It may rise above the
   original 1.3% (readable merged operands, larger direct share, trained
   operand embeddings) — the claim requires only direct ≪ CoT, and the
   width sweep predicts one-step-with-merged-retrieval fails at dim 16.
   *Refuted if* direct accuracy approaches CoT.
3. Sampled format mix ≈ 50/50, tracking the training mixture.
4. The carry probe on the fully re-tokenized replay — now in-distribution —
   still reads near chance for the deep (2-step) carry at the answer
   position. This is the load-bearing prediction: if the probe now recovers
   the carry, the §1 "false negative for interpretability" claim was partly
   an artifact of distribution shift, and we will say so.
