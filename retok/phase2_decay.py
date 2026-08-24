"""How far forward does a non-canonical span contaminate the replay?

Tokenization damage is **localized** — BPE pre-tokenizes on whitespace, so a
non-canonical split of one word does not change how later words are encoded.
But **causal damage is not localized**: attention is causal, so in the real
generation every subsequent position attended to the *actual* tokens. Re-tokenize
the transcript and every later position now attends to a token that was never
there. The question is how fast that washes out.

For each non-canonical span we walk forward k tokens and compare, at the same
*text* position:

- ``actual``: the model's own token prefix, continued k tokens past the span.
- ``replay``: the canonical re-encoding of that identical text.

Both decode to the same string; they differ only in how the earlier span was
segmented (plus any length offset it induces). The KL between their next-token
distributions at distance k is the residual causal contamination.

A fast decay means the damage is confined to the span's neighbourhood. A slow
decay means a single non-canonical token silently perturbs the rest of the
transcript — which is the case that matters for per-position interpretability,
since it would mean *every* later position is analyzed under a counterfactual
context.

Run:
    uv run python -m retok.phase2_decay \
        --model meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import difflib

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.gpu import resolve_device
from retok.phase2_probe import (
    PROMPTS,
    decode_for_roundtrip,
    roundtrip_is_measurable,
)

# Distances (in actual-stream tokens) past the end of the non-canonical span.
DISTANCES = [0, 1, 2, 4, 8, 16, 32, 64]


def _first_replace(
    actual: list[int], canon: list[int]
) -> tuple[int, int, int, int] | None:
    if actual == canon:
        return None
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=actual, b=canon, autojunk=False
    ).get_opcodes():
        if op != "equal":
            return i1, i2, j1, j2
    return None


@torch.no_grad()
def _next_dist(model: object, ids: list[int], device: torch.device) -> torch.Tensor:
    x = torch.tensor([ids], device=device)
    return F.softmax(model(x).logits[0, -1].float(), dim=-1)  # type: ignore[attr-defined]


@torch.no_grad()
def run(
    model_name: str,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> None:
    device = resolve_device(require_cuda=False)
    print(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map={"": device.type}, low_cpu_mem_usage=True
    )
    model.eval()
    torch.manual_seed(seed)
    use_chat = tokenizer.chat_template is not None

    # distance -> list of (kl, flipped)
    by_dist: dict[int, list[tuple[float, bool]]] = {d: [] for d in DISTANCES}
    n_spans = 0

    for prompts in PROMPTS.values():
        for prompt in prompts:
            text0 = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if use_chat
                else prompt
            )
            enc = tokenizer(text0, return_tensors="pt").to(device)
            plen = enc.input_ids.shape[1]
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
            )
            for seq in out:
                gen = seq[plen:].tolist()
                while gen and gen[-1] in (
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                ):
                    gen.pop()
                if len(gen) < 8:
                    continue
                text = decode_for_roundtrip(tokenizer, gen)
                if not roundtrip_is_measurable(tokenizer, text):
                    continue
                canon = tokenizer.encode(text, add_special_tokens=False)
                region = _first_replace(gen, canon)
                if region is None:
                    continue
                _, i2, _, _ = region
                n_spans += 1
                for d in DISTANCES:
                    end = i2 + d
                    if end >= len(gen):
                        break
                    prefix = gen[:end]
                    ptext = decode_for_roundtrip(tokenizer, prefix)
                    if not roundtrip_is_measurable(tokenizer, ptext):
                        continue
                    replay = tokenizer.encode(ptext, add_special_tokens=False)
                    if replay == prefix:
                        continue  # re-canonicalized away; no contamination left
                    p_a = _next_dist(model, prefix, device)
                    p_c = _next_dist(model, replay, device)
                    kl = float(
                        (
                            p_a
                            * (p_a.clamp_min(1e-12).log() - p_c.clamp_min(1e-12).log())
                        ).sum()
                    )
                    by_dist[d].append((kl, int(p_a.argmax()) != int(p_c.argmax())))

    print("\n===== CAUSAL CONTAMINATION vs DISTANCE PAST THE SPAN =====")
    print(f"model: {model_name}   non-canonical spans analyzed: {n_spans}")
    print(f"{'distance':>9}{'n':>7}{'meanKL':>9}{'medKL':>9}{'top1 flip':>11}")
    for d in DISTANCES:
        vals = by_dist[d]
        if not vals:
            continue
        kls = torch.tensor([v[0] for v in vals])
        flip = sum(v[1] for v in vals) / len(vals)
        print(
            f"{d:>9}{len(vals):>7}{kls.mean():>9.3f}{kls.median():>9.3f}{flip:>10.0%}"
        )
    print(
        "\nInterpretation: distance 0 is the boundary metric reported elsewhere.\n"
        "Non-zero KL at large distance = the re-tokenized replay is still running\n"
        "a different computation many tokens later, because attention is causal\n"
        "and the earlier position differs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal contamination decay")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
