# Results: retok (hidden computational composition)

> **Note on this log.** This is the run log from the private monorepo this
> experiment was extracted from — an authentic record rather than a tidied one,
> including the dead-ends and self-corrections. Two mechanical edits were made:
> module paths were rewritten (`experiments.retok.X` → `retok.X`) to match this
> repo, and one broken markdown table header was repaired. Nothing else was
> changed, so the commands below are in the monorepo's idiom. Map them as
> follows:
>
> | in this log | in this repo |
> |---|---|
> | `./train.sh local retok -- ARGS` | `uv run python -m retok.train ARGS` |
> | `RUN_NAME=...` / `--save-checkpoint` | no public equivalent — the final checkpoint is always written to `--checkpoint-dir` |
> | `s3://brendanlong-experiments/retok/checkpoints/<run>/final.pt` | `hf:checkpoints/<run>/final.pt` in the [HF dataset](https://huggingface.co/datasets/brendanlong/retok-noncanonical-tokenization) |
> | `s3://brendanlong-experiments/retok/artifacts/<f>.jsonl` | `<f>.jsonl` at the root of the same dataset |
> | `skypilot/train-retok.yaml`, `skypilot/probe-retok*.yaml` | [`skypilot/reproduce.yaml`](skypilot/reproduce.yaml) (generic) |
>
> The private S3 bucket is not public; the HF dataset above is the public
> equivalent. wandb run IDs are in project `retok` under the `brendanlong`
> entity. Reproduction entry points are in [`scripts/`](scripts/).


**Question.** Can a model's actual token stream differ from the canonical
re-tokenization of its output in a way that flips per-position interpretability
and monitoring analysis — and can we demonstrate it cleanly in a controlled
toy? See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for hypothesis and
pre-registered predictions.

## Release

Published as a standalone public repo on 2026-08-17:
**https://github.com/brendanlong/tokenization-hidden-computation-experiment**
(flattened layout — `retok.*` → `retok.*`, the used subset of
`shared/` vendored as `common/`; S3 upload and run-name guards stripped).

Artifacts are in the public HF dataset
[`brendanlong/retok-noncanonical-tokenization`](https://huggingface.co/datasets/brendanlong/retok-noncanonical-tokenization):
the per-model generation records at the root, and **all 17 checkpoints** under
`checkpoints/<run_name>/final.pt` — so every `s3://` checkpoint URI below has a
public equivalent, `hf:checkpoints/<run_name>/final.pt`.

## Runs

> Each entry MUST record: exact copy-pasted command, model config, batch size,
> LR, schedule, GPU, wandb run ID, and a result summary (per-format,
> per-carry-chain-length accuracy; direct-mode P(correct)).

### Official Phase-1 runs (SkyPilot local-gpu, RTX 3060, wandb, S3 checkpoints)

Model (all runs): decoder-only, 2 layers, 4 heads, learned PE, tied embeddings,
GELU MLP. Base-10, 3-digit reversed addition. Batch 512, LR 3e-4 cosine, 200
warmup, streaming single-epoch unique data. Eval is in-distribution (see
`make_eval_buckets` — intentional). Config defaults reproduce the headline with
no overrides.

**Headline — dim=16 uncued mixture (CoT + single merged token), 3 seeds.**
Command (seed s ∈ {0,1,2}):
```
RUN_NAME=retok-main-s$s ./train.sh local retok -- --seed $s --save-checkpoint --wandb-run-name retok-main-s$s
```
- Config: dim=16, `direct_fraction`=0.3, `generate_n`=20M (39k steps).
- wandb runs: `gb6a5vp4` (s0), `2zhdhlkl` (s1), `wrlumw7d` (s2), project `retok`.
- Checkpoints: `s3://brendanlong-experiments/retok/checkpoints/retok-main-s{0,1,2}/final.pt`.

| metric | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| CoT accuracy | 100.0% | 100.0% | 99.8% | **99.9%** |
| direct (single merged token) | 1.1% | 1.5% | 1.4% | **1.3%** |
| direct one-step P(correct) | 0.003 | 0.003 | 0.003 | **0.003** |

Per carry-chain length (s0): CoT 100/100/100%; direct 1.3/0.7/1.3% — flat across
chains (the failure is per-position arithmetic capacity, not carry depth).

**Reconciling the two direct-mode numbers** (they are conditioned differently and
look inconsistent side by side; 3-seed means, 4000 pairs):

| quantity | value |
|---|---|
| restricted-argmax accuracy, *within the merged subspace* | 1.27% |
| unconditional P(correct merged token) | 0.0027 |
| total probability mass on the merged branch | 0.289 |
| **conditional** P(correct \| merged branch) | 0.0092 |
| full-vocab argmax picks *any* merged token | **0.00%** |

0.0027 / 0.289 = 0.0093, so the two published figures agree once conditioning is
matched: a token holding ~0.9% of the branch mass wins a 1000-way argmax ~1.3% of
the time. Note the model **never** prefers the one-token form to starting the
digit-by-digit answer (full-vocab argmax 0.00%), and the merged mass tracks
`direct_fraction`=0.3 as trained. The calibration detector uses the
*unconditional* number, which is correct — it asks how probable the transcript is
as a one-step generation.

**Analysis** (`retok.analysis` on `retok-main-s0`, 2000 pairs/bucket):
- **Positions collapse:** real CoT answer = **3** decode positions; canonical
  replay = **1** (structural — by construction of the encoding + `canonicalize()`,
  pinned by `test_reversed_convention_and_canonical_collapse`, not a fitted
  measurement). Serial-depth / per-position accounting on the stored transcript
  undercounts decode steps 3→1.
- **Calibration detector:** re-tokenized correct transcripts are 100% correct,
  yet the model assigns each merged answer only **P ≈ 0.002** in one step —
  astronomically unlikely as one-pass generation (text + logprobs only).
- **Carry probe** (best layer, per digit; base rate ~51%; both sides probed
  pre-emission — CoT's D dedicated computing-positions vs the replay's single
  `=` position). Carry into digit 1 (= `a0+b0 ≥ 10`, a function of the two
  visible low operand digits) is fully recoverable at `=` on **both** streams
  (CoT 95.6%, replay 100%). The deeper 2-step carry into digit 2 is recovered
  better at CoT's dedicated computing-position (**81.8%**) than crammed at the
  single `=` (**68.8%**) — the positional-dedication benefit only shows up for
  the carry that needs the chain. Against a replay that keeps the operands as
  readable digit tokens, then, the information **survives** re-tokenization
  (carries are operand functions) and this is a positional-structure result.

  **But that replay is not what a stored transcript becomes** (2026-08-17).
  A real encoder runs over the whole string, so the operand runs merge too:
  `<bos> 7 5 0 + 8 6 0 = 5 2 1` re-encodes to `<bos> [750] + [860] = [521]`,
  13 positions → 7. Probed on *that* sequence the carry signal drops to
  **66.2%** (digit 1) and **54.5%** (digit 2) against a 50.9% base — near
  chance.

  **This is not information loss, and calling it that would be wrong.** The
  merged tokens are a *bijection* on 0..999, so `[750]` and `[860]` still
  determine the operands exactly and every carry is recoverable from the
  re-tokenized token IDs with no model involved at all. What breaks is that
  **this model's residual stream no longer linearly encodes the carry** at those
  positions — it never learned to unpack a merged operand token, having only
  ever seen merged tokens after `=`. That is an interpretability-validity
  result, which is the claim we actually want: a probe run on the stored
  transcript reports near-chance for a carry the model demonstrably computed.
  A false negative, not a vanished fact. Both replays are reported: the
  digit-operand
  replay isolates *capacity* (can it one-step, given a readable prompt?), the
  fully re-tokenized one is *what an analyst actually holds*. Caveat: the latter
  is off-distribution, since merged tokens only follow `=` in training — which
  is itself the point, a re-tokenized transcript is a sequence the model never
  emitted and cannot consume.

**Appendix — width sweep (dim 8–64, 2 layers, seed 0, 15M examples).** Locates
the dim=16 regime and shows CoT tolerates far more width-compression than the
one-pass forms.
```
RUN_NAME=retok-sweep-cot-d$d   ./train.sh local retok -- --dim $d --seed 0 --generate-n 15000000 --wandb-run-name retok-sweep-cot-d$d
RUN_NAME=retok-sweep-1step-d$d ./train.sh local retok -- --mode one_step --dim $d --seed 0 --generate-n 15000000 --wandb-run-name retok-sweep-1step-d$d
```

| dim | CoT | one-step (parallel per-digit, retrieval-free) | direct (merged token) |
|----:|----:|----:|----:|
| 8  | 0.9% | 1.0% | 0.3% |
| 12 | 89.8% | 9.9% | 0.9% |
| **16** | **100%** | **10.5%** | **1.5%** |
| 24 | 100% | 100% | 2.0% |
| 32 | 100% | 59.5%* | 3.8% |
| 64 | 100% | 100% | 97.5% |

Ordering of required capacity: **CoT (dim≥12–16) < one-step compute (dim≥24) <
one-step + merged-token retrieval (dim≈64)**. dim=16 is the unique window where
only CoT works. At dim=64 the merged-token form succeeds (97.5%) — wider models
*can* one-pass it, which is why the claim is scoped to this model at dim=16.
(*dim=32 one-step 59.5% is a single-seed outlier between the 100%s at dim=24/64;
reseed pending.)

**Why width, not depth:** depth=1 breaks CoT itself (D=3, L=1 CoT ≈ 9% in the
sweeps) — a CoT step must compute a digit and expose its outgoing carry, wanting
~2 layers. So 2 layers is the CoT floor and width is the only free knob.

## Related work checkpoint (Aug 2026)

"Weird Re-Tokenization, symmetries and compression: research agenda"
(Xenomirant & Sami Wolf, LessWrong, 17 Aug 2026) surveys this area from an
alignment perspective. Assessment: **not scooped**, but a close neighbour.

> **Posted a summary comment on that post, 17 Aug 2026:**
> <https://www.lesswrong.com/posts/osWWL4yfentdhX9Q7/weird-re-tokenization-symmetries-and-compression-research?commentId=siCkKxAkPCZBmuCEo>
>
> Covers the toy (a transcript that appears to show one-step 3-digit addition
> from a model too small to do it), the wild rates falling with scale, and the
> prompted-induction result, and links the public repo. Flags two planned
> follow-ups: a harder search for prompts that induce non-canonical output
> (including on larger models), and training a model to hide computation via RL
> against a monitor that reads the **re-tokenized** text. Watch the thread for
> replies from the authors — their agenda names *writer invariance* as an open
> question, which §3 partly answers.

- It is a *survey + agenda*, not a results post.
- **Overlaps**: wild rates (well covered by the literature it cites — Geh et al.
  2025 on canonicity vs generation length, plus Chatzi and Vieira which we
  already cite); and a decay-with-distance result (1.9–3.2× odds the next word is
  non-canonical, falling to 1.35× by ~30 tokens). That last one is the closest
  hit, but it measures **non-canonicality propagating**, whereas our decay curve
  measures the **next-token distribution shifting** (KL / argmax flip). Same
  shape, different quantity.
- **Does not cover**: our toy, the calibration detector, and — most usefully —
  prompted induction. The agenda names *writer invariance* ("does the model
  assign significant probability to non-canonical token sequences?") as a **new,
  unexplored question**, and flags "can the model steer itself... while evading
  CoT monitors?" as the single bullet its own preliminary experiments have not
  answered. Our induction result answers the first directly.
- **Useful new citations it surfaced**: Kaplan et al. (2025), early-layer
  "detokenization" collapsing subword runs into a word-level vector — a
  mechanistic account of why our contamination stops at the word boundary;
  Ghosh & Jyothi (2026), invariance preserved for English but degrading for
  fragmented languages — corroborates our language effect; Jain et al. (2026),
  `retok@k`; Sims et al. (2026) / Steger et al. (2026) on stochastic-retokenization
  finetuning.

## Phase 2 — wild-caught non-canonical generation

Establishes that the phenomenon is real in deployed models, unprompted, no
training. Detection is mechanical: a generation is non-canonical iff
`encode(decode(gen_ids)) != gen_ids` — you just retain the actual token IDs
(logprobs expose them for closed APIs). Detector verified against a hand-built
non-canonical split. Local RTX 3060, small open models, temp 1.0, sampled.
Code: `retok.phase2_probe` (rate + gallery),
`retok.phase2_interp` (interp mismatch).

> **TWO CORRECTIONS supersede earlier versions of this section.**
>
> **(a) Qwen is not immune.** An early run reported Qwen2.5 at 0% in every
> domain and we recommended tokenizer choice as a mitigation. That was a
> small-sample fluke (0 of 64). Qwen generates non-canonically with genuine
> BPE-merge spans (`' Car' 'ry'` → `' Carry'`); the recommendation is withdrawn.
>
> **(b) The first cross-model comparison was confounded.** Three harness bugs,
> all pushing in the direction of our headline, are now controlled:
> - **`clean_up_tokenization_spaces`** ships **True for Llama-3.2** and False for
>   GPT-2/Qwen. It destructively rewrites `" ."→"."`, `" 's"→"'s"` and eight
>   more, producing spurious non-canonical detections *for Llama only*.
> - **Sampling params were inherited, not pinned.** HF takes anything unset from
>   the model repo's `generation_config.json`, which differs per model (Qwen2.5:
>   `top_k=20, top_p=0.8, repetition_penalty=1.1`; Llama-3.2: `top_p=0.9`;
>   GPT-2: none). Off-canonical tokens live in the tail, so this suppressed Qwen
>   relative to GPT-2 by ~3×.
> - **We also passed `top_p=0.95` ourselves**, truncating the tail for everyone.
> - Added filters for **truncated UTF-8 (U+FFFD)** and **Qwen's NFC normalizer**,
>   both of which cause false positives.
>
> All numbers below are the **controlled** re-measurement.

**Scale ladder — the rate falls with model size** (Llama family, identical
tokenizer, controlled harness, temp 1.0, per-token rate):

| model | english | arithmetic | code | ml-Latin | ml-Cyrillic | ml-CJK |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-**1B** | 0.03% | 0.12% | 0.35% | 1.28% | 4.06% | 5.20% |
| Llama-3.2-**3B** | 0.00% | 0.08% | 0.05% | 0.99% | 1.98% | 2.93% |
| Llama-3.1-**8B** | 0.03% | 0.03% | 0.03% | 0.42% | 1.31% | 1.21% |

<sub>The 8B row was stale until 2026-08-18 (it read 0.00 / 0.07 / 0.11 / 0.31 /
1.47 / 1.38, from a pre-fp32 run). Every cell above is now recomputed from the
published token-ID artifacts and agrees with the FINAL table below.</sub>

**Monotone decreasing in scale in every domain** — CJK falls 4.3×, Cyrillic
3.1×, Latin 3.0×, code 12× from 1B to 8B. (The previously noted "one exception,
code 3B < 8B" was an artifact of the stale row; the true code trend is
0.35% → 0.05% → 0.03%.) Holding the tokenizer *and* the precision fixed isolates
this as a *model* effect, consistent with the tail-sampling mechanism:
better-trained models concentrate more mass on the canonical continuation,
leaving less tail to sample from.

**Across families the direction holds but is noisier** (all domains pooled,
per-token): GPT-2 124M **1.66%**, Llama-3.2-1B **0.96%**, Qwen2.5-1.5B
**0.24%**, Gemma-2-2b **0.24%**, Llama-3.2-3B **0.52%**, Llama-3.1-8B **0.26%**,
gpt-oss-20b **0.03%**. Smallest and largest are 55× apart, but the middle is not
ordered — Qwen-1.5B and Gemma-2b both sit below Llama-3B. So the *controlled*
claim is the within-family ladder; the cross-family version is a trend, not a
law. Both panels are in `figures/phase2_scale.png`.

### Scale, restated on Latin script (2026-08-18)

The scale figure was per-prompt-domain, which the section below shows is
confounded, and it omitted the three comparison models. Both fixed by measuring
**Latin-script tokens only** — the one slice every model actually produces, and
where n is largest (~21k–38k tokens per model). `figures/phase2_scale.png`.

| model | params | Latin rate | english prompt | non-english prompt |
|---|---:|---:|---:|---:|
| GPT-2 | 0.12B | 1.92% | 1.00% | 2.06% |
| Llama-3.2-1B | 1.24B | 1.33% | 0.03% | 1.65% |
| Qwen2.5-1.5B | 1.54B | 0.28% | 0.00% | 0.35% |
| Gemma-1-2B | 2.51B | 0.54% | 0.00% | 0.74% |
| Gemma-2-2b | 2.61B | 0.24% | 0.05% | 0.27% |
| Llama-3.2-3B | 3.21B | 0.72% | 0.00% | 0.88% |
| Gemma-3-4B | 4.30B | 0.07% | 0.00% | 0.09% |
| Llama-2-7B | 6.74B | 0.10% | 0.00% | 0.13% |
| Llama-3.1-8B | 8.03B | 0.30% | 0.03% | 0.36% |
| gpt-oss-20b | 20.9B | 0.03% | 0.00% | 0.04% |

**Within-family it is monotone**, on two independent ladders: Llama-3.x
1.33% → 0.72% → 0.30%, and Gemma generations 0.54% → 0.24% → 0.07%. The Gemma
one is as much a *recency* ladder as a size one (2.5B → 2.6B → 4.3B), which
suggests training recipe matters at least as much as parameter count.

**Across families it is not a scaling law.** Llama-3.2-3B (0.72%) sits above
Qwen2.5-1.5B (0.28%); Llama-2-7B (0.10%) sits below Llama-3.1-8B (0.30%). The
endpoints span 64× (GPT-2 1.92% → gpt-oss-20b 0.03%) but the middle does not
order. Report the within-family claim; do not draw a global trend line.

**English-only is a floor, not an alternative.** Every model after GPT-2 is
0.00–0.05% on Latin tokens written for English prompts, so that slice cannot
discriminate between models — the entire signal lives in Latin tokens written in
response to non-English prompts, which is the off-distribution mechanism again.
This is why "restrict to English" is not the fix for the domain confound and
"restrict to Latin" is.

**Residual confound, not removable from this data.** Latin fixes *which script*
is measured but not what the model chose to write in it. Llama-2-7B answers
everything in English (10% in-script on Cyrillic prompts), so its Latin tokens
are more routine text than Llama-3.1-8B's, which include code-switching around
genuine Russian and Japanese output. That plausibly explains its position below
the Llama-3.x line, and it means cross-family cells still are not fully
like-for-like.

### Direct comparison: their models, our methodology (2026-08-18)

The "Weird Re-Tokenization" agenda post's fig. 2 orders models the opposite way
we do, so we measured the three of its five that fit an A40 with **our** setup
(200 max new tokens, temp 1.0, pure sampling, as-released dtype, n=12/prompt —
identical to the seven models above). Mamba-130M skipped (not a transformer),
Qwen3-30B-A3B skipped (needs an A100-80GB).

Command: `sky launch skypilot/probe-retok.yaml -c retok-lw --infra runpod
--gpus A40:1 --down --yes --retry-until-up --env RUN_NAME=lw-compare --env
PROBE_MODELS="google/gemma-2b-it google/gemma-3-4b-it
meta-llama/Llama-2-7b-chat-hf"`. ~11 min on an A40 at $0.44/hr ≈ **$0.10**.
Artifacts: `lw_comparison/` in the HF dataset.

| model | pooled per-token | per-generation | their fig. 2 (per-seq @512, uncond.) |
|---|---:|---:|---:|
| Gemma-1-2B-it | **0.62%** | 25% | ~1.5% — their **lowest** |
| Llama-2-7B-chat | 0.10% | 6% | ~58% |
| Gemma-3-4B-it | **0.04%** | 3% | ~72% — their **highest** |

**The ordering is inverted on the two Gemmas.** We find Gemma-1-2B the *least*
canonical of the three and Gemma-3-4B the *most*, by ~15×; they find the
reverse, by ~48×. Ours is what our scale/recency story predicts — Gemma-3 (2025,
4B) concentrates far more mass on the canonical continuation than Gemma-1 (2024,
2B). Theirs is what makes their size plot non-monotone.

**Not a script-composition artifact.** Both Gemmas answer in the prompt's script
(Gemma-1-2B 77% Cyrillic / 93% CJK; Gemma-3-4B 59% / 97%), so the confound from
the section above does not explain the gap. By output script:

| model | Latin | Cyrillic | CJK |
|---|---:|---:|---:|
| Gemma-1-2B-it | 0.54% (21516) | **4.55%** (1603) | 2.11% (2082) |
| Gemma-3-4B-it | 0.07% (25289) | **0.10%** (2079) | 0.00% (2968) |
| Llama-2-7B-chat | 0.10% (32417) | 3.93% (382) | 0.00% (561) |

45× apart on Cyrillic tokens, same direction. (Llama-2-7B answers Russian and
Japanese prompts in English — 10% / 12% in-script — so its multilingual cells
are thin and mostly measure Latin text, like the small Llama-3.2s.)

**Assessment.** Two independent disagreements with that figure now: the sign of
the length trend (ours rises, theirs falls, and non-recovering BPE requires
rising), and the ordering of the two Gemmas. Both are large. We cannot explain
either from the figure alone, so the likeliest reading is that their y-axis
measures a different quantity than `encode(decode(ids)) != ids` on the emitted
tokens. Worth resolving in the thread before either set of numbers is cited; not
presented as a refutation.

### The domain labels are the PROMPT's language, not the output's (2026-08-18)

Prompted by "could the rates differ because models talk about different
things?" — they can, and it changes the mechanism. Every domain rate above is
keyed by the prompt. Measuring what the models actually *wrote* (fraction of
letters in the expected script):

| model | ml-Cyrillic prompt | ml-CJK prompt |
|---|---:|---:|
| Llama-3.2-1B | 11% Cyrillic | 7% CJK |
| Llama-3.2-3B | 11% Cyrillic | 12% CJK |
| Llama-3.1-8B | **82%** Cyrillic | **37%** CJK |
| Qwen2.5-1.5B | 89% Cyrillic | 97% CJK |

The small Llamas answer Russian and Japanese prompts overwhelmingly **in
English**. So "Llama-3.2-1B, ml-Cyrillic, 4.06%" is a rate over text that is 89%
*not* Cyrillic, and comparing that cell to Llama-3.1-8B's compares different
content as much as different models.

Re-attributing every token to the script of its own surface
(`phase2_script.py`, CPU-only, ≥90% of non-canonical tokens are letter-bearing
so the view is near-complete):

| model | Latin | Cyrillic | CJK |
|---|---:|---:|---:|
| GPT-2 | 1.92% (38511) | 0.32% (624) | n<300 |
| Llama-3.2-1B | 1.33% (31504) | 0.86% (350) | 0.42% (708) |
| Qwen2.5-1.5B | 0.28% (30145) | 0.15% (2671) | 0.76% (3419) |
| Gemma-2-2b | 0.24% (30126) | 1.16% (1546) | 1.07% (1688) |
| Llama-3.2-3B | 0.72% (32675) | 1.33% (525) | 0.29% (1026) |
| Llama-3.1-8B | 0.30% (31507) | 1.24% (2499) | 0.12% (2433) |
| gpt-oss-20b | 0.03% (35134) | 0.28% (2127) | 0.00% (2232) |

**CJK tokens are among the *lowest*-rate tokens, not the highest.** The headline
"CJK up to 5.2%" is a property of CJK-*prompted* generations, not of CJK text.

**What actually drives it — the prompt pushing the model off-distribution.**
Holding the script fixed at Latin and splitting by which prompt produced it:

| model | english prompt | ml-CJK prompt | ml-Cyrillic prompt |
|---|---:|---:|---:|
| Llama-3.2-1B | 0.03% (6220) | **6.91%** (2360) | **4.81%** (1830) |
| Llama-3.1-8B | 0.03% (6133) | **3.36%** (1189) | **2.18%** (367) |
| GPT-2 | 1.00% (5190) | 1.53% (1826) | 0.40% (2263) |

Same script, same model — a **~200× higher** rate on Latin tokens written in
response to a CJK prompt than to an English one. So the effect is not "some
scripts tokenize badly"; it is **"text produced off-distribution is sampled from
a flatter distribution, and non-canonical tokens live in the tail"** — the same
tail-sampling mechanism as the temperature result, reached by a different route.
This is a better story than the one we had, and it is the one the data supports.

**The scale result survives this, controlled.** Within Latin script it is
monotone for the Llama ladder (1.33% → 0.72% → 0.30%, n≈32k each — the
best-powered version of the claim), and the off-distribution rates fall the same
way (CJK-prompted Latin 6.91% → 3.36%; Cyrillic-prompted 4.81% → 2.18%). Only
the *by-script Cyrillic* column fails to be monotone (0.86% → 1.33% → 1.24%),
and its 1B cell rests on 350 tokens. So: report the scale claim on Latin script
or pooled, not on the multilingual domain columns.

### Generation length is a first-order confound for sequence-level rates

All rates here are at **200 `max_new_tokens`** (mean ~180 actually emitted).
That matters, because the *per-generation* rate is a strong function of length
while the per-token rate is not. Truncating the published generations to N
tokens and re-running the round trip:

| | N=32 | N=64 | N=128 | N=200 |
|---|---:|---:|---:|---:|
| GPT-2 — per-generation | 20% | 34% | 53% | **64%** |
| GPT-2 — per-token | 1.58% | 1.57% | 1.72% | **1.67%** |
| Llama-3.2-1B — per-generation | 7% | 18% | 27% | **33%** |
| Llama-3.2-1B — per-token | 0.52% | 0.88% | 0.94% | **1.05%** |
| Qwen2.5-1.5B — per-generation | 3% | 6% | 10% | **16%** |
| Qwen2.5-1.5B — per-token | 0.23% | 0.25% | 0.20% | **0.21%** |

Per-generation rises ~3× over this range and saturates at 100%; per-token is
roughly flat. This follows from BPE being **non-recovering** (Chatzi et al.):
once a sequence goes off-canonical every extension of it stays off-canonical, so
the sequence-level flag can only accumulate. **A sequence-level rate quoted
without its length is therefore uninterpretable**, and two such rates at
different lengths are not comparable. Figure: `figures/phase2_length.png`.

This **contrasts with Vieira et al. (ICML 2025)**, who report non-monotone
scaling (their Llama-3.1-8B worse than Llama-3.2-3B). The setups differ — they
measure unconditional canonicality at 1024 tokens, we measure prompted
generation at 200 — so this is a difference to flag rather than a refutation.

It also **contrasts with fig. 2 of the "Weird Re-Tokenization" agenda post**,
which reports sequence-level non-canonicality *falling* with sampled
continuation length (e.g. Gemma-1-2B from ~90% at 32 tokens to ~0% at 1024).
Ours rises with length for every model, which is the direction non-recovering
BPE requires. We have not been able to reconcile the two: their y-axis is
"percent of non-canonical token sequences", which should be monotonically
non-decreasing in length under the same definition we use, so the difference is
likely a different quantity rather than a different finding. Worth resolving
before either number is cited — flagged in the comment thread.
Their model set is also not a scale ladder (Mamba-130M, Gemma-1-2B, Gemma-3-4B,
Llama-2-7B, Qwen3-30B-A3B span four architectures and several generations), so
their size ordering is confounded with training recipe in a way ours is not.

**FINAL cross-family table — 7 models, as-released precision** (`--dtype auto`,
i.e. how each model is normally run), pure sampling at temp 1.0, per-token rate.
Every cell is recomputable on CPU from the published token-ID artifacts via
`retok.phase2_verify` (all 7 verified exact).

| model | dtype | english | arith | code | ml-Latin | ml-Cyrl | ml-CJK |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-2 (124M) | fp32 | 0.89% | 1.41% | 1.54% | **3.28%** | 0.43% | 1.09% |
| Llama-3.2-1B | bf16 | 0.03% | 0.12% | 0.35% | 1.28% | **4.06%** | **5.20%** |
| Qwen2.5-1.5B | bf16 | 0.00% | 0.05% | 0.08% | 0.76% | 0.25% | 0.59% |
| Gemma-2-2b | fp32 | 0.07% | 0.06% | 0.12% | 0.45% | 0.66% | 0.54% |
| Llama-3.2-3B | bf16 | 0.00% | 0.08% | 0.05% | 0.99% | 1.98% | 2.93% |
| Llama-3.1-8B | bf16 | 0.03% | 0.03% | 0.03% | 0.42% | 1.31% | 1.21% |
| gpt-oss-20b | bf16* | 0.03% | 0.00% | 0.00% | 0.11% | 0.14% | 0.00% |

<sub>*gpt-oss ships MXFP4; the MXFP4 kernels need compute capability ≥9.0
(Hopper) and no H100 capacity was available on RunPod across 28 region attempts,
so it ran bf16-dequantized on an A100. This is the one cell that is **not**
as-released precision.</sub>

- **gpt-oss-20b is the most canonical model tested** — 0.00% on arithmetic,
  code and CJK; only English and the two non-Latin scripts register at all.
- **Scale ladder** (Llama family, tokenizer *and* dtype held fixed): CJK
  5.20% → 2.93% → 1.21% and Latin 1.28% → 0.99% → 0.42% from 1B → 3B → 8B.
  Monotone in every multilingual domain; the sub-0.1% English/code cells are at
  the noise floor.
- **Precision is not a free parameter.** Measuring Gemma-2 at its as-released
  fp32 rather than bf16 moved every cell (CJK 0.94% → 0.54%), and GPT-2 likewise
  (Cyrillic 1.09% → 0.43%). A rate quoted without the dtype is underspecified,
  exactly like one quoted without the sampling parameters.

**Reproducing.** Generation needs a GPU; *analysis* does not:

```bash
# regenerate (GPU)
uv run python -m retok.phase2_probe --model <name> --dtype auto \
    --n-samples 12 --max-new-tokens 200 --temperature 1.0 --seed 31 \
    --jsonl-out data/retok/artifacts/<name>.jsonl

# re-derive every rate from the published token IDs (CPU only)
aws s3 sync s3://brendanlong-experiments/retok/artifacts/ data/retok/artifacts/
uv run python -m retok.phase2_verify data/retok/artifacts/*.jsonl
```

Artifacts (~900KB/model, ~300 records each) carry the token IDs the model
**actually emitted**, the canonical re-encoding, the decoded text, the span
diffs, and the full run config. `phase2_verify` recomputes canonicality from the
IDs rather than trusting the stored flags, so it audits the analysis
independently of generation — it is what caught a dtype regression that had
silently shifted GPT-2's numbers.

**How the per-token rate is attributed.** Detection is sequence-level
(`encode(decode(ids)) != ids`), so turning it into a per-token rate needs an
explicit choice. It is **not** the boolean divided by the token count. The two
sequences are aligned and the emitted tokens inside a changed region are counted
(`phase2_probe.py:150`, `:320`):

```python
canon = tokenizer.encode(tokenizer.decode(gen_ids))
for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
    a=gen_ids, b=canon, autojunk=False
).get_opcodes():
    if op != "equal":
        n_bad += i2 - i1  # tokens on the ACTUAL side of the region
rate = n_bad / len(gen_ids)
```

So the quantity is *"what fraction of the tokens the model actually emitted sat
in a span that re-tokenization altered"* — which is the right denominator for
interp validity, since those are exactly the positions whose residuals an
analyst would be misreading.

Two implementation details that matter more than they look:

- **`autojunk=False` is load-bearing.** `SequenceMatcher` by default treats any
  element occurring in >1% of a sequence of length ≥200 as "junk" and will not
  anchor matches on it. On token streams that is spaces, `the`, punctuation —
  precisely the long generations we care about would get a distorted diff.
- **`SequenceMatcher` is not a minimal edit distance**, it is a recursive
  longest-contiguous-match heuristic. For the short localized substitutions here
  it agrees with a minimal alignment, but that is not guaranteed in general.
  Pure-insertion opcodes (assigned 0, since no emitted token was touched) are
  negligible: 2 of 419 differing opcodes for GPT-2, 0 for both Llamas.

**Sensitivity to the choice** — alternatives move the number ~±20%, so the
qualitative claims do not rest on it. The naive version is 5× off *and*
reintroduces the length confound, being algebraically per-generation ÷ mean
length:

| model | ours (actual side) | canonical side | max(both) | naive bool/len |
|---|---:|---:|---:|---:|
| GPT-2 | **1.66%** | 1.36% | 1.77% | 0.33% |
| Llama-3.2-1B | **0.96%** | 1.08% | 1.22% | 0.17% |
| Llama-3.1-8B | **0.26%** | 0.30% | 0.34% | 0.08% |

**Both metrics, and the identity that reconciles them.** Reporting the
sequence-level number alongside is a useful cross-check, because the two are
related exactly by

```
per_token = per_generation × (mean affected tokens per non-canonical generation)
                           ÷ (mean tokens per generation)
```

Verified to hold to floating point for all ten models measured
(`test_per_token_identity` in `tests/`):

| model | per-generation | mean len | affected/non-canon gen | per-token |
|---|---:|---:|---:|---:|
| GPT-2 | 60% | 183 | 5.0 | 1.66% |
| Llama-3.2-1B | 29% | 170 | 5.5 | 0.96% |
| Qwen2.5-1.5B | 16% | 184 | 2.7 | 0.24% |
| Gemma-2-2b | 15% | 167 | 2.6 | 0.24% |
| Llama-3.2-3B | 22% | 180 | 4.2 | 0.52% |
| Llama-3.1-8B | 15% | 182 | 3.2 | 0.26% |
| gpt-oss-20b | 3% | 196 | 2.0 | 0.03% |
| Gemma-1-2B-it | 25% | 142 | 3.5 | 0.62% |
| Gemma-3-4B-it | 3% | 180 | 2.5 | 0.04% |
| Llama-2-7B-chat | 6% | 182 | 3.1 | 0.10% |

The middle column is the interesting one: a non-canonical generation carries
**2–5.5 affected tokens**, not one. That is why per-generation over-states and
the naive division under-states — a single flag stands in for a small cluster of
tokens, and the cluster size varies by model (GPT-2 5.0, gpt-oss-20b 2.0).

**Metric note.** The **per-token** rate is primary. BPE is *non-recovering*
(Chatzi et al.): once a sequence goes non-canonical, every extension of it stays
non-canonical. So the per-*generation* rate rises monotonically with generation
length and **saturates at 100%** — Llama's "100% on CJK" means every one of 18
rollouts contained ≥1 span, not that every token was affected (its per-token rate
there is 5.20%). Per-generation numbers are therefore only comparable at fixed
length, and are reported below in parentheses for continuity with earlier runs.

**Rate table — pure sampling, temperature 1.0** (`top_k=0, top_p=1.0,
repetition_penalty=1.0`, `clean_up_tokenization_spaces=False`,
transformers 4.57.6, ~60–96 generations per cell, seed 31). % of generations
with ≥1 non-canonical span, and per-token rate:

| domain | GPT-2 | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---:|---:|---:|
| english | 49% (1.30%/tok) | 3% (0.03%) | **0%** (0.00%) |
| arithmetic | 68% (1.82%) | 5% (0.12%) | 5% (0.05%) |
| code | 55% (1.25%) | 18% (0.35%) | 8% (0.08%) |
| multilingual — Latin | **89%** (3.51%) | 56% (1.28%) | 38% (0.76%) |
| multilingual — Cyrillic | 36% (1.09%) | 71% (4.06%) | 17% (0.25%) |
| multilingual — CJK | 67% (1.45%) | **100%** (5.20%) | 48% (0.59%) |

Findings:

- **Rates are far higher under pure sampling than with tail truncation.** Our
  earlier (confounded, `top_p=0.95`) table read 6–33%; with the full tail
  exposed the same models reach 49–89% (GPT-2) and up to 100% (Llama, CJK). This
  is the single largest methodological lesson here: **this metric is a function
  of the sampling configuration**, so a rate quoted without the full sampling
  parameters is meaningless.
- **The model ordering is consistent** across every domain: GPT-2 > Llama-3.2 >
  Qwen2.5, on both per-generation and per-token rates. Qwen really is the most
  canonical of the three — just not immune.
- **Language dominates.** English is the *low* extreme for the modern models
  (Llama 3%, Qwen 0%) while multilingual is the high extreme (Llama 56–100%).
  Reporting a single "multilingual" number, as we did before, hides a 3× spread
  between scripts.
- **CJK is not structurally immune, contrary to expectation.** The prior
  reasoning was that CJK is ~1 char/token so alternatives don't exist — true
  enough that we split the bucket, but Llama-3.2 is **100%** non-canonical on
  CJK, so its vocabulary clearly contains multi-character CJK tokens with
  competing segmentations.
- The **English** column is where the models differ most (49% / 3% / 0%), which
  is also the only cell where an "essentially never happens" claim is defensible
  — and only for the two modern models.

**What actually drives the rate: sampling temperature** (`phase2_temperature`,
controlled, code + Latin-multilingual prompts, 64 generations per cell):

| temperature | Qwen2.5-1.5B | Llama-3.2-1B |
|---|---:|---:|
| 0.0 (greedy) | **0%** | **0%** |
| 0.7 | 9% (0.22%/tok) | 14% (0.22%) |
| 1.0 | 20% (0.42%) | 41% (0.95%) |
| 1.5 | 92% (3.33%) | 91% (3.68%) |
| 2.0 | 95% (3.77%) | 88% (4.30%) |

Non-canonical generation is overwhelmingly a **tail-sampling phenomenon**: both
models are **exactly 0% under greedy decoding** and climb to ~90% of generations
by temperature 1.5. Temperature matters more than model identity or tokenizer —
the Qwen/Llama gap at a fixed temperature (roughly 2×) is small next to the
0%→90% swing across the temperature range.

*Caveat:* at temp ≥1.5 many generations are excluded by the U+FFFD filter (the
model emits enough garbage to truncate mid-character), so those cells rest on
20–43 measurable generations rather than 64, and may be biased toward the
better-formed samples.
This is consistent with the mechanism being "the model concentrates mass on the
canonical continuation, and non-canonical tokens are sampled from the tail" —
not "some models can't do it." (The temp-0.7 vs 1.0 non-monotonicity in both
columns is n=48 noise; the trend across the full range is not.)

The digit-tokenization observation **does** survive: Qwen tokenizes digits
individually, and its arithmetic-domain spans are words (`' Carry'`, `' Ones'`),
never numbers — so there is no *digit* channel in Qwen, even though the subword
channel is open.

- **The per-generation rate overstates it; the per-*token* rate is sub-1%.**
  Llama-3.2-1B non-canonical tokens as a fraction of all generated tokens:
  english **0.00%**, arithmetic **0.04%**, code **0.54%**, multilingual
  **0.85%**. A stored transcript is ~99%+ token-identical to what ran — the
  effect is sparse, concentrated at a handful of positions per completion
  (≤~0.8 spans/generation).
- **Tokenizer ambiguity is NOT the differentiator.** Counting distinct
  segmentations per word via lattice DP, Qwen2.5 and Llama-3.2 are **identical**
  (594, 1212, 278, 5153, 280 for a five-word probe set); GPT-2 is slightly lower
  (535, 1120, 256, 3957, 272) despite having the *highest* rate. Every tokenizer
  tested offers ample non-canonical paths — the differences between models are
  behavioural (how sharply mass concentrates on the canonical path), not
  structural.
- **Numbers are a genuine per-tokenizer difference.** Llama-3.2 emits numbers
  *canonically* (a focused check found **0/12** digit-involving spans; its
  arithmetic hits are subword/punctuation). Qwen tokenizes digits individually so
  it has no digit channel at all (its arithmetic spans are words like `' Carry'`,
  `' Ones'`). **GPT-2 does emit non-canonical numbers** (` 5010`, ` 8101`) — the
  digit-mirror to the toy exists in older models.

**Interp mismatch — re-tokenizing changes the *actual* computation, not just the
token count** (`phase2_interp`, controlled, Llama-3.2-1B, **600 generations, 149
non-canonical boundaries**, seed 41, temp 1.0, 200 new tokens). At the text
boundary right after a non-canonical span, compare the model's next-token
distribution under the actual vs the canonical re-tokenization of the *identical
prefix text*:

- the top-1 next token **flips for 51% of boundaries** (76/149, 95% Wilson CI
  **43–59%**).
- but the flips are **overwhelmingly low-confidence**: at flipped boundaries the
  actual run's median probability on its *own* top-1 was **0.09** — it was
  near-indifferent — and 0.01 on the token the replay picks. Requiring the
  actual run to have been confident (p>0.5) leaves **4% of boundaries**
  (6/149, 95% CI **2–9%**). So ~92% of the flips are the two tokenizations
  breaking a near-tie differently.
- next-token **KL = 0.75 nats mean, 0.335 median** (max 5.67). Canonical control
  0 by construction.
- per domain (boundaries / flip% / confident-flip%): multilingual-Latin 61 / 56%
  / 5%; multilingual-Cyrillic 25 / 64% / 4%; multilingual-CJK 35 / 49% / 0%;
  code 23 / 30% / 9%; english 4 / 25% / 0%; arithmetic 1 / 100% / 0% (n=1).

### Greedy decoding is NOT 0% — correction (2026-08-17, `phase2_probe --temperature 0`)

The earlier "exactly 0% under greedy decoding" came from `phase2_temperature`,
whose prompt set is `PROMPTS["code"][:4] + PROMPTS["multilingual_latin"][:4]` —
**8 prompts, 1 deterministic generation each, 2 models**. Greedy is
deterministic, so `--n-samples` cannot add evidence there; only more prompts and
more models can. Re-ran at argmax over the **full 25-prompt set**:

Command (per model):

```
uv run python -m retok.phase2_probe --model <name> --dtype auto \
    --n-samples 1 --max-new-tokens 200 --temperature 0.0 --seed 31 \
    --jsonl-out data/retok/greedy/<name>.jsonl
```

(`phase2_probe` previously hardcoded `do_sample=True`; it now switches to
`do_sample=False` at temperature 0 and forces 1 generation per prompt.)

| model | gens with ≥1 span | per-token | where |
|---|--:|--:|---|
| GPT-2 | 2/24 | 0.12% | arithmetic (2/5 prompts) |
| Llama-3.2-1B | 1/25 | 0.08% | code |
| Qwen2.5-1.5B | 2/25 | 0.09% | Cyrillic, code |
| Llama-3.2-3B | 1/25 | 0.05% | multilingual-Latin |
| **pooled** | **6/99** | **0.08%** | |

**So the model's own argmax continuation is sometimes a non-canonical token.**
The original 8-prompt subset contained no arithmetic prompts at all, which is
precisely where GPT-2's greedy hits are — a sampling-of-prompts artifact, not a
sampling-of-tokens one. Temperature still dominates (0.08% → 0.4–1.0% → 3–4%),
but it buys ~an order of magnitude, not immunity, and "turn the temperature
down" is not a mitigation.

Gemma-2-2b was skipped: as-released fp32 at 2B does not fit the 8 GB local card.
Llama-3.1-8B and gpt-oss-20b likewise need a bigger GPU.

### Can a prompt *induce* non-canonical generation? (`phase2_induce`)

Everything above is accidental — tokens sampled from the tail. This asks whether
a model can be **instructed** into a specific non-canonical segmentation, which
is what separates "validity bug" from "potential channel". Deliberately *not*
tested here: mid-token prompt boundaries and non-canonically-tokenized context.
Both would show that *we* can confuse the model by manipulating its input, which
is already known; here the input is ordinary text and the model must choose the
segmentation itself.

| probe | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|
| **concat** — "write `light` immediately followed by `house`, no separator" | **93%** induced (38/41) | **75%** (6/8) |
| **repeat** *(control)* — "repeat this word exactly: `lighthouse`" | **0%** (0/8) | **0%** (0/8) |
| **question** *(control)* — word never shown; "what is the tall coastal tower…?" | **0%** (0/3) | 17% (1/6) |
| **spell** — "one character at a time, no separators" | 0% (1 produced) | **0%** (8/8 produced) |
| **digits** — "one digit at a time, no separators" | **0%** (0/30) | n/a (digit-individual) |

("induced" requires the model to emit the target string *and* the tokens
covering it to be non-canonical; producing the right text canonically is an
induction failure — that is the distinction the metric exists for.)

**The controls are what make this interpretable.** Without them, 93% induction
could simply mean these compounds are *always* emitted in pieces, with the prompt
doing nothing. On the same words: asking the model to **repeat** the word gives
canonical output 16/16 times across both models, and asking a **question whose
answer is the word** — never showing it — gives canonical output 8 of 9 times.
Pooled, controls are **1/25 (4%)** non-canonical against concat's **44/49
(90%)**; Fisher exact p ≈ 1.5×10⁻¹³, odds ratio ≈ 211. The prompt is doing the work, not the word.

*Caveat:* the question control has low power — models often answered with a
sentence rather than the bare word, leaving only 3 (Llama) and 6 (Qwen) usable
trials. The repeat control (16/16 canonical) carries most of the weight.

**Prompted induction works, but only through semantics, not form.**

- **concat succeeds almost every time the model complies**: `['light']['house']`
  where canonical is `['l']['ighthouse']`; `['key']['board']` where canonical is
  the *single* token `['keyboard']`; `['ins']['ide']` vs `['inside']`. The
  compliance rate is modest (42% / 15% — models often add quotes or explanation)
  but conditional on complying, the segmentation is non-canonical ~100% of the
  time.
- **spell and digits fail completely.** When the model does emit the bare target,
  it emits it canonically. Instructing a change to the *format of a single unit*
  ("one character at a time", "one digit at a time") does not reach the token
  stream at all.

The mechanism this implies: **the model has no handle on its own tokenization**,
so instructions about form don't touch it. What works is giving it a *semantic*
reason to treat the output as two things — the segmentation follows the concepts,
not the instruction. That also means the `digits` probe, the direct wild analogue
of our toy, **fails**: you cannot simply ask a model to emit a number
digit-by-digit at the token level.

**Why this matters for §"the reason to care before it matters".** It converts the
RL argument from structural speculation into something with a demonstrated
mechanism: a policy would not need to learn anything about tokenization to move
into non-canonical space — it would only need to learn to *conceptualise* its
output in pieces, which prompting already shows is sufficient. It is
simultaneously mild evidence for the low-infohazard read: the available handle is
coarse (word-level concatenation), not a controllable encoder.

Artifacts: `induce_<model>.jsonl` alongside the others, one record per trial.

**Causal contamination decays fast in magnitude but leaves a persistent flip
tail** (`phase2_decay`, Llama-3.2-1B, 49 non-canonical spans). Tokenization
damage is *localized* — BPE pre-tokenizes on whitespace, so a bad split of one
word does not change how later words encode, and the Chatzi "non-recovering"
property only means the sequence-level **flag** stays true, not that
mis-tokenization spreads. But **causal** damage is not localized: attention is
causal, so every later position in the replay attends to a token that was never
emitted. Walking k tokens past the span (comparing at matched *text* positions):

| distance past span | n | mean KL | median KL | top-1 flip |
|---:|---:|---:|---:|---:|
| 0 (the boundary metric) | 48 | 0.531 | 0.227 | 50% |
| 1 | 49 | 0.201 | 0.062 | 27% |
| 2 | 49 | 0.087 | 0.025 | 14% |
| 4 | 49 | 0.138 | 0.016 | 20% |
| 8 | 48 | 0.054 | 0.007 | 12% |
| 16 | 46 | 0.013 | 0.004 | 9% |
| 32 | 44 | 0.012 | 0.003 | 11% |
| 64 | 41 | 0.022 | 0.002 | 15% |

- **Magnitude dilutes ~57× within 16 tokens** (median KL 0.227 → 0.004),
  and ~110× by 64 tokens (→ 0.002).
- **But the top-1 flip rate does not reach zero** — it falls from 50% to a
  plateau around **9–15%** and stays there out to 64 tokens. A re-tokenized
  transcript is still picking a different argmax at roughly 1 in 8 positions,
  arbitrarily far downstream of the span that caused it.
- **Honest reading of the plateau:** at those distances the two distributions
  are nearly identical (median KL 0.002), so the persistent flips are *near-ties*
  — the same coin-flip phenomenon as the boundary analysis, not a substantively
  different computation. The plateau plausibly reflects the base rate of
  near-ties in LM distributions being resolved differently by a tiny
  perturbation, rather than lasting corruption.

So the correct statement is: **a single non-canonical token perturbs every
subsequent position, but the perturbation is tiny after ~16 tokens.** For
per-position interpretability this means the *span neighbourhood* is genuinely
unreliable, while distant positions are only affected where the model was
near-indifferent anyway.

**Comparison to the confounded run** (top_p=0.95, Llama decode post-processing
on): it reported 29% flips / 10% confident. Exposing the full tail roughly
doubles the raw flip rate (29% → 51%) while *halving* the confident-flip rate
(10% → 4%) — every additional flip the tail exposes is a near-tie. The
"mostly soft" conclusion is therefore **stronger** under proper controls, not
weaker.

Concrete boundaries (actual span | canonical span → predicted next token):

- `[' с' 'олн']` | `[' солн']` → `'еч'` vs `'ца'` (Russian *солнеч…* vs *солнца*)
- `[' m' 'ue']` | `[' mue']` → `'g'` vs `'bles'` (Spanish, → *muebles*)
- `['[' '^.']` | `['[^' '.']` → `']*'` vs `'*/'` (regex; KL 3.3) — semantic
- `['zeit' 'ig']` | `['zeitig']` → `' wurde'` vs `' hat'` (German)

**Honest magnitude:** real and measurable, but per-instance **small and mostly
soft** — sub-1% of tokens, and most next-token flips are low-confidence near-ties
or whitespace, with a thin tail (**4%** of the sparse boundaries, 6/149,
CI 2–9%) of
confident changes, of which only a minority are semantic. **Framing:** the toy carries the controlled mechanism +
the hidden-computation/calibration claim; Phase 2 is the existence proof. The
case for logging token IDs rests on *silent + near-zero mitigation cost + tail
risk on a span you can't identify in advance*, not on magnitude. Example
galleries: `phase2_probe` / `phase2_interp` stdout.

### Go/no-go sweeps (local RTX 3060, direct `uv run`, results discarded)

> **Superseded — do not cite the numbers in this section.** These are short
> undertrained probes (6–8M examples) that preceded the official 15M sweep. Where
> the two disagree, the official sweep (appendix table above) is authoritative:
> re-evaluating the **published** checkpoints on 2026-08-17 reproduces the
> appendix row-for-row (dim=24 one-step 100% / direct 1.9%; dim=32 one-step
> 59.1% / direct 3.4%; dim=64 one-step 100% / direct **97.7%**), not the numbers
> below. In particular the "Bonus (dim=64): direct-merged = 1%" claim does **not**
> hold for the official runs — at dim=64 the merged form succeeds, which is
> exactly why the headline claim is scoped to dim=16. The dim=32 one-step 59.5%
> also *replicates* (59.1%), so it is a property of that checkpoint rather than
> an eval fluke — though whether it is a training-seed outlier is still untested.

A sequence of short local sweeps to find a task regime where CoT succeeds while
direct one-step addition fails with carry-chain length. These drove three
design corrections and one substantive finding about the task itself.

**1. Merged-operand encoding → 0% everywhere.** First sweep (`n_layers ∈
{1,2,4}`, base-10, D=4, operands as merged tokens) gave 0% CoT at every depth
(token acc stuck at the digit-marginal ~34%). Cause: a merged operand token
hides the digits the model must add. **Fix:** operands as individual digit
tokens. With that, 2 layers reach ~100% CoT.

**2. Full-vocab argmax metric conflated format choice with correctness.** The
CoT first digit and the direct merged token are both predicted at `=`, so with
a 50/50 mixture the merged token stole the argmax and CoT read as ~75% (D=2)
even when arithmetic was perfect. **Fix:** per-format accuracy uses argmax
restricted to that format's token subspace; calibration P(correct) stays
full-softmax.

**3. base-10 forces retrieval-vs-chain-length tension.** With the metric fixed
(dim 256, L=2, `direct_fraction` 0.3):

| base | D | merged vocab | CoT | direct (argmax) | direct P(correct) |
|---|---|---|---|---|---|
| 10 | 3 | 1,000 | 100% | ~60% (flat) | 0.12→0.10 (mild grad) |
| 10 | 4 | 10,000 | 100% | ~0% (flat) | ~0.00 (flat) |

D=4's failure is **retrieval** (10⁴-way readout won't train at this scale), not
a serial-depth limit — a vocab artifact, not a real "one-step is impossible"
result. D=3's chains are too short (direct one-step ~60%). **Fix:** make the
numeric base a knob — a small base gives long carry chains with a small,
trainable merged vocab.

**4. Finding: binary addition is too parallelizable (shortcut to constant
depth).** base-2, D=8 (vocab 256, chains 0–7), dim 128, `direct_fraction` 0.3:

CoT = 100% and **direct = ~100% at every chain length, including length 7**, at
just 2 layers. Addition is TC⁰ (carry-lookahead is parallelizable), so a shallow
transformer resolves the whole chain in one step — the "Transformers Learn
Shortcuts to Automata" effect the plan flagged for solvable groups. `direct
P(correct)` ≈ 0.30 flat = the format-choice mass (`≈ direct_fraction`), not a
capability limit. So the one-step-*impossible* punchline does not hold for
addition per se; it needs either a model too shallow for the carry-lookahead
depth (D large relative to layers) or a non-parallelizable task. Current test:
base-2, D=12 (vocab 4096, chains 0–11) at L∈{2,3} — does shallow depth fail the
*long* chains (a genuine depth limit) while CoT succeeds?

**5. Result: no — addition never shows a depth-limited one-step failure.**
base-2, D=12, dim 256, L=2, `direct_fraction` 0.3, 24M examples: CoT = 100% at
every chain length; direct argmax ≈ **23% flat** across chains 0–11 (chain_0=22%,
chain_7=17%, chain_11=30% — no downward trend), `direct P(correct)` ≈ 0.025 flat.
Flat-across-chains is the signature of a **retrieval** limit (constant difficulty
per example), not a **depth** limit (which would fall as the carry chain
lengthens). Across the whole sweep, direct-mode accuracy is always flat in chain
length — 100% (binary D=8), 23% (binary D=12), 60% (base-10 D=3), 0% (base-10
D=4) — the level tracks merged-vocab size (retrieval), never carry-chain length
(serial depth). **Conclusion: addition is parallelizable (TC⁰); growing D to
make one-step fail hits the retrieval wall first.**

**6. Resolution: shrink the *model*, not the problem — width sweep at D=3.**
The clean separation comes from making the model the bottleneck at small,
trainable D=3 (base-10, vocab 1000). Two levers:
- *Depth to 1 layer:* breaks CoT too (D=3, L=1: CoT ≈ 9%). A CoT step must
  compute the digit *and* expose the outgoing carry — wants ~2 layers. So 1
  layer removes the thing we compare against.
- *Width at L=2:* one-step must compute all D digits at the `=` position (its
  capacity split D ways), while CoT dedicates a position/forward-pass to each
  digit. Narrowing therefore breaks one-step before CoT.

Width sweep, base-10 D=3, L=2, 15M examples (accuracy = all D digits correct):

| dim | CoT | one-step (parallel per-digit) | direct (merged token) |
|----:|----:|----:|----:|
| 64 | 100% | 100% | 1% |
| 32 | 100% | 99% | 4% |
| 24 | 100% | 100% | 3% |
| **16** | **100%** | **10.5%** | 1.5% |

At **dim=16** CoT = 100% but one-step = 10.5%, and the *parallel* one-step
(no merged-vocab retrieval) fails too — so the failure is computation/capacity,
not retrieval. one-step fails specifically on the units digit (pos 0 ≈ chance,
pos 1–2 ≈ 99%): capacity split D ways drops the hardest digit; CoT gives each
digit its own position. one-step stays ~10% from 6M→15M examples (a capacity
floor, not undertraining), while dim-16 CoT rises 1%→100% (was undertrained).

**Bonus (dim=64):** one-step-parallel = 100% but direct-merged = 1% — same
computation, only the output representation differs, so this *isolates* the
merged-token failure as pure retrieval. Confirms the direct-merged control
conflates computation with output-vocab; the parallel readout is the clean
capability measure.

**Chosen design:** base-10, D=3, L=2, a **width sweep (dim 8–64)** as the
headline (CoT tolerates far more compression than one-step), with a canonical
**dim≈16** model — CoT 100%, one-step ≈10% — for the retok probe/replay/
calibration analysis. Small, trainable vocab; no binary, no large D, no S5.

<!-- template
#### n_layers=2

Command:
```
RUN_NAME=retok-sweep-2L ./train.sh local retok -- --n-layers 2 --generate-n 8000000 --wandb-run-name retok-sweep-2L
```
- Model: 2L / 128d / 4h, vocab 10015, seq_len 10, tied embeddings, learned PE
- Batch size: 512 | LR: 3e-4 cosine, 200 warmup | direct_fraction: 0.15
- GPU: local RTX 3060 (SkyPilot local-gpu)
- wandb: <id>
- Result: CoT mean __%; direct mean __%; per chain_len {0:__, 1:__, 2:__, 3:__};
  direct P(correct) per chain_len {…}
-->
