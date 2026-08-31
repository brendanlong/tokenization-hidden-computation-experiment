---
license: mit
task_categories:
  - text-generation
language:
  - en
  - de
  - es
  - fr
  - pt
  - ru
  - ja
  - zh
tags:
  - tokenization
  - interpretability
  - bpe
size_categories:
  - 1K<n<10K
---

# Non-canonical tokenization in LLM generations

Per-generation records from seven language models, capturing **the token IDs each
model actually emitted** alongside the canonical re-encoding of its own output —
plus the trained toy-model checkpoints from the accompanying controlled
experiment.

Code, writeup and full run log:
**https://github.com/brendanlong/tokenization-hidden-computation-experiment**

Tokenization is many-to-one: many token sequences decode to the same string, but
re-encoding that string recovers only the *canonical* sequence. So storing a
transcript as text and re-tokenizing it preserves the string while silently
changing the segmentation — and any analysis that operates on positions (logit
lens, probes, per-position attribution, serial-depth accounting) then runs on
positions the model never produced.

This dataset exists so that the rates reported in the accompanying writeup can be
**recomputed on CPU, without re-running generation**.

## Models

| model | tokenizer lineage | dtype (as released) |
|---|---|---|
| `gpt2` | GPT-2 BPE | fp32 |
| `meta-llama/Llama-3.2-1B-Instruct` | tiktoken-derived | bf16 |
| `meta-llama/Llama-3.2-3B-Instruct` | tiktoken-derived | bf16 |
| `meta-llama/Llama-3.1-8B-Instruct` | tiktoken-derived | bf16 |
| `Qwen/Qwen2.5-1.5B-Instruct` | Qwen BBPE | bf16 |
| `google/gemma-2-2b` | SentencePiece | fp32 |
| `openai/gpt-oss-20b` | o200k (OpenAI) | bf16\* |

<sub>\*gpt-oss ships MXFP4, whose kernels need Hopper; this ran bf16-dequantized
on an A100. The only model not at as-released precision.</sub>

Generation settings: temperature 1.0, **pure sampling** (`top_k=0, top_p=1.0,
repetition_penalty=1.0` — pinned explicitly, not inherited from each repo's
`generation_config.json`), 200 max new tokens, seed 31, `transformers` 4.57.6,
`clean_up_tokenization_spaces=False`.

## Contents

| path | what |
|---|---|
| `<model>.jsonl` | per-generation records backing the rate table |
| `induce_<model>.jsonl` | prompted-induction trials + matched controls |
| `greedy/<model>.jsonl` | greedy-decoding (temperature 0) runs over the full prompt set |
| `lw_comparison/<model>.jsonl` | the three "Weird Re-Tokenization" comparison models, our methodology |
| `api/<model>.jsonl` | frontier models measured via chat-completions logprobs (OpenAI: `generated_ids`; OpenRouter: `sampled_tokens` strings + `provider`) |
| `api/induce_<model>.jsonl` | the prompted-induction probe + controls on the frontier models |
| `api_pinned/<model>.jsonl` | re-run of the API measurements with sampling truncation pinned explicitly (`top_p=1`; OpenRouter also `top_k=0` where endpoints accept it) — see RESULTS.md "API re-measurement" |
| `rl/<run>.rollouts.jsonl` | per-rollout records from the RL arms (token IDs, targets, per-eval; gpt2-large expansion replication + Qwen2.5-3B reversal) |
| `interp_<model>.jsonl` | boundary-divergence records (per-generation, with KL / next-token IDs at the first non-canonical boundary) |
| `decay_<model>.jsonl` | contamination-decay records (per-generation, with KL / top-1 flip at each distance past the span) |
| `temperature_<model>.jsonl` | temperature-sweep records (rate-table schema plus varying `temperature`) |
| `checkpoints/retok-main-s{0,1,2}/final.pt` | the 3 original toy-model seeds (2 layers, dim 16; uncued 70/30 mixture) |
| `checkpoints/retok-mergedops-s{0,1,2}/final.pt` | the merged-operands variant (50/50; direct format fully canonical incl. operands; s2's CoT undertrained — see RESULTS.md) |
| `checkpoints/retok-sweep-cot-d<N>/final.pt` | width sweep, CoT arm, dim ∈ {8,12,24,32,64} |
| `checkpoints/retok-sweep-1step-d<N>/final.pt` | width sweep, one-step arm, dim ∈ {8,12,16,24,32,64} |
| `checkpoints/retok-sweep-{1L,2L,4L}/final.pt` | depth sweep |
| `checkpoints/retok-main-s<seed>/final.pt.json` | metadata sidecar (config + final eval) |

20 runs in total. The CoT arm has no `d16` entry because the dim=16 CoT model
*is* `retok-main-s{0,1,2}` — that is the headline run. Sidecars exist only for
those three; the sweep runs predate the sidecar.

Checkpoints are `{"step", "model_state_dict", "model_config"}` dicts, loadable
with `torch.load(..., weights_only=True)`, and resolve by run name from the code:

```bash
uv run python -m retok.analysis --checkpoint hf:checkpoints/retok-main-s0/final.pt
```

Run names match RESULTS.md 1:1: `retok-main-s<seed>` are the headline runs,
`retok-sweep-cot-d<N>` / `retok-sweep-1step-d<N>` are the width sweep that
located the dim=16 regime (digit-by-digit succeeds, one pass fails).

## Schema

One JSON object per generation (`<model>.jsonl`):

| field | meaning |
|---|---|
| `model` | model id |
| `domain` | `english`, `arithmetic`, `code`, `multilingual_latin`, `multilingual_cyrillic`, `multilingual_cjk` |
| `prompt` | the user prompt |
| `seed`, `temperature` | generation config |
| `generated_ids` | **the token IDs the model actually emitted** |
| `canonical_ids` | `encode(decode(generated_ids))` — what a stored-as-text transcript becomes |
| `text` | decoded output |
| `non_canonical` | whether `canonical_ids != generated_ids` |
| `spans` | per differing region: `surface`, `actual` tokens, `canonical` tokens |
| `excluded`, `exclude_reason` | rounds trips that can't be measured (`replacement_char` for truncated UTF-8, `non_nfc` for Qwen's NFC normalizer) |

`induce_<model>.jsonl` holds the prompted-induction trials, with `probe`,
`target`, `produced`, `induced`, `actual_tokens`, `canonical_tokens`.

`interp_*.jsonl` adds a `boundary` object per generation (`region`, `kl`,
`actual_next_id`, `canonical_next_id`, `pa_top1`, `pa_at_canon`);
`decay_*.jsonl` adds `measurements` (per distance: `kl`, `flip`). The token-ID
bookkeeping in both is recomputable on CPU (`--from-jsonl` in the matching
module); the stored KL/probability values are what a GPU re-run would
regenerate.

Provenance note: the `interp_*` / `decay_*` / `temperature_*` files are pinned
re-runs of the original sweeps (which predate artifact publishing); the
writeup quotes these files, and the original runs' slightly different values
are preserved in the repo's RESULTS.md errata.

## Verifying

```python
import json
from transformers import AutoTokenizer

rows = [json.loads(l) for l in open("meta-llama_Llama-3.2-1B-Instruct.jsonl")]
tok = AutoTokenizer.from_pretrained(rows[0]["model"])

bad = tot = 0
for r in rows:
    if r["excluded"]:
        continue
    ids = r["generated_ids"]
    text = tok.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    canon = tok.encode(
        text, add_special_tokens=False
    )  # recompute, don't trust the flag
    if canon != ids:
        bad += sum(
            i2 - i1
            for op, i1, i2, _, _ in __import__("difflib")
            .SequenceMatcher(a=ids, b=canon, autojunk=False)
            .get_opcodes()
            if op != "equal"
        )
    tot += len(ids)
print(f"non-canonical tokens: {bad / tot:.2%}")
```

Or use the repo's verifier, which does this across every file and checks the
recomputed values against the stored flags:

```bash
uv run python -m retok.phase2_verify *.jsonl
```

## Gotchas if you measure this yourself

- **Pin the sampling parameters.** HF inherits anything unset from the model
  repo's `generation_config.json`, and these differ per model. Off-canonical
  tokens live in the distribution tail, so top-k/top-p truncation suppresses the
  metric — unevenly across models.
- **Disable `clean_up_tokenization_spaces`.** It ships `True` for Llama-3.x and
  `False` for GPT-2/Qwen, and destructively rewrites `" ." → "."`, `" 's" → "'s"`
  and eight more, producing spurious non-canonical detections for one family only.
- **Report the dtype.** This is a tail-sampling measurement and precision moves
  it: GPT-2 at bf16 rather than native fp32 shifts its Cyrillic rate 0.43% → 1.09%.
- **Exclude unmeasurable round trips**: outputs containing U+FFFD (truncated
  UTF-8 can't re-encode) and, for Qwen, non-NFC text (its tokenizer normalizes on
  encode).

## Source

Code, full run log and writeup:
https://github.com/brendanlong/tokenization-hidden-computation-experiment
