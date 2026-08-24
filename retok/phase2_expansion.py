"""Non-canonical rate on LONG digit runs (decimal expansions).

The published rate table's arithmetic prompts elicit short numbers, and no
modern model emitted a single non-canonical digit span there. Long decimal
expansions are the case where we have a mechanistic reason to expect otherwise:
BPE canonicality on a digit run is not prefix-consistent, so a left-to-right
generator cannot always stay canonical.

Measured on GPT-2, appending one digit at a time to random 14-digit runs:

    tokenizer      prefix stable   only last token re-chunks   earlier token changes
    GPT-2                     4%                         77%                     19%
    Llama-3.2-1B             33%                         67%                      0%
    Qwen2.5-1.5B            100%                          0%                      0%

So for GPT-2, in ~19% of steps staying canonical would require *revising an
already-emitted token* — impossible for an autoregressive model. The other 77%
is decidable only if the model already knows where it will stop. Llama's 3-digit
left grouping is prefix-consistent (0% deeper churn); Qwen emits one digit per
token, so there is nothing to re-chunk.

Note this measures **tokenization, not correctness**: a model emitting 20 wrong
digits still answers "can it tokenize a long digit run canonically?".

Run:
    uv run python -m retok.phase2_expansion --model gpt2 --places 20
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.gpu import resolve_device
from retok.phase2_probe import (
    decode_for_roundtrip,
    diff_spans,
    roundtrip_is_measurable,
)

# Divisors chosen so the expansion is long and non-repeating early, and so the
# fraction is not a famous constant a model might have memorised.
DIVISORS = [7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]


def expansion(a: int, b: int, places: int) -> str:
    r, out = a % b, []
    for _ in range(places):
        r *= 10
        out.append(str(r // b))
        r %= b
    return "".join(out)


def prompts(places: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    for b in DIVISORS:
        a = rng.randint(1, b - 1)
        out.append(
            (
                f"Write {a}/{b} as a decimal to {places} places. "
                f"Output only the digits after the decimal point, nothing else.",
                expansion(a, b, places),
            )
        )
    return out


@torch.no_grad()
def run(
    model_name: str,
    places: int,
    n_samples: int,
    temperature: float,
    seed: int,
    dtype: str,
) -> None:
    device = resolve_device(require_cuda=False)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto" if dtype == "auto" else getattr(torch, dtype),
        device_map={"": device.type},
        low_cpu_mem_usage=True,
    ).eval()
    torch.manual_seed(seed)

    agg = defaultdict(int)
    examples: list[str] = []
    for prompt, truth in prompts(places, seed):
        text = prompt
        if tok.chat_template:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=places + 24,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            num_return_sequences=n_samples,
            pad_token_id=tok.eos_token_id,
        )
        for seq in out:
            ids = seq[enc.input_ids.shape[1] :].tolist()
            while ids and ids[-1] in (tok.eos_token_id, tok.pad_token_id):
                ids.pop()
            gen = decode_for_roundtrip(tok, ids)
            if not roundtrip_is_measurable(tok, gen):
                agg["excluded"] += 1
                continue
            agg["gens"] += 1
            agg["tokens"] += len(ids)
            longest = max((len(r) for r in _digit_runs(gen)), default=0)
            agg["digits_emitted"] += longest
            res = diff_spans(tok, ids)
            if res is None:
                continue
            _, spans = res
            digit_spans = [s for s in spans if any(c.isdigit() for c in s.surface)]
            if spans:
                agg["noncanon_gens"] += 1
            if digit_spans:
                agg["gens_with_digit_span"] += 1
                agg["digit_spans"] += len(digit_spans)
                for s in digit_spans[:1]:
                    if len(examples) < 4:
                        examples.append(f"{s.actual_tokens} | {s.canonical_tokens}")
            agg["correct"] += int(truth[:8] in gen)

    g = max(1, agg["gens"])
    print(f"\n===== DECIMAL EXPANSION, {places} places =====")
    print(
        f"model={model_name}  temp={temperature}  n={agg['gens']} "
        f"(excluded {agg['excluded']})"
    )
    print(f"  mean longest digit run emitted : {agg['digits_emitted'] / g:.1f}")
    print(
        f"  generations non-canonical      : {agg['noncanon_gens']}/{g} "
        f"= {agg['noncanon_gens'] / g:.0%}"
    )
    print(
        f"  with a DIGIT span              : {agg['gens_with_digit_span']}/{g} "
        f"= {agg['gens_with_digit_span'] / g:.0%}   "
        f"({agg['digit_spans']} spans total)"
    )
    print(
        f"  first 8 digits correct         : {agg['correct']}/{g} "
        f"= {agg['correct'] / g:.0%}   (capability, not the metric)"
    )
    for e in examples:
        print(f"    {e}")


def _digit_runs(text: str) -> list[str]:
    runs, cur = [], ""
    for ch in text:
        if ch.isdigit():
            cur += ch
        elif cur:
            runs.append(cur)
            cur = ""
    if cur:
        runs.append(cur)
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--places", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--dtype", default="auto")
    a = p.parse_args()
    run(a.model, a.places, a.n_samples, a.temperature, a.seed, a.dtype)


if __name__ == "__main__":
    main()
