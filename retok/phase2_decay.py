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

Run (GPU):
    uv run python -m retok.phase2_decay \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --jsonl-out data/retok/artifacts/decay_meta-llama_Llama-3.2-1B-Instruct.jsonl

Re-derive the table from saved records (CPU only; span regions and replay
identity are recomputed from the raw IDs, the KL/flip values are read from the
records since recomputing them needs the model):

    uv run python -m retok.phase2_decay \\
        --from-jsonl hf:decay_meta-llama_Llama-3.2-1B-Instruct.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

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


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _report(
    model_name: str, n_spans: int, by_dist: dict[int, list[tuple[float, bool]]]
) -> None:
    print("\n===== CAUSAL CONTAMINATION vs DISTANCE PAST THE SPAN =====")
    print(f"model: {model_name}   non-canonical spans analyzed: {n_spans}")
    print(f"{'distance':>9}{'n':>7}{'meanKL':>9}{'medKL':>9}{'top1 flip':>11}")
    for d in DISTANCES:
        vals = by_dist.get(d, [])
        if not vals:
            continue
        kls = [v[0] for v in vals]
        flip = sum(v[1] for v in vals) / len(vals)
        print(
            f"{d:>9}{len(vals):>7}{sum(kls) / len(kls):>9.3f}"
            f"{_median(kls):>9.3f}{flip:>10.0%}"
        )
    print(
        "\nInterpretation: distance 0 is the boundary metric reported elsewhere.\n"
        "Non-zero KL at large distance = the re-tokenized replay is still running\n"
        "a different computation many tokens later, because attention is causal\n"
        "and the earlier position differs."
    )


def run(
    model_name: str,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    jsonl_out: Path | None = None,
) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common.gpu import resolve_device
    from retok.phase2_probe import (
        PROMPTS,
        decode_for_roundtrip,
        roundtrip_is_measurable,
    )

    device = resolve_device(require_cuda=False)
    print(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map={"": device.type}, low_cpu_mem_usage=True
    )
    model.eval()
    torch.manual_seed(seed)
    use_chat = tokenizer.chat_template is not None

    @torch.no_grad()
    def _next_dist(ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], device=device)
        return F.softmax(model(x).logits[0, -1].float(), dim=-1)

    # distance -> list of (kl, flipped)
    by_dist: dict[int, list[tuple[float, bool]]] = {d: [] for d in DISTANCES}
    records: list[dict[str, object]] = []
    n_spans = 0

    with torch.no_grad():
        for domain, prompts in PROMPTS.items():
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
                    rec: dict[str, object] = {
                        "model": model_name,
                        "domain": domain,
                        "prompt": prompt,
                        "seed": seed,
                        "temperature": temperature,
                        "generated_ids": gen,
                        "measurements": [],
                    }
                    records.append(rec)
                    text = decode_for_roundtrip(tokenizer, gen)
                    if not roundtrip_is_measurable(tokenizer, text):
                        rec.update(excluded=True, non_canonical=False)
                        continue
                    canon = tokenizer.encode(text, add_special_tokens=False)
                    rec.update(excluded=False, canonical_ids=canon)
                    region = _first_replace(gen, canon)
                    rec["non_canonical"] = region is not None
                    if region is None:
                        continue
                    _, i2, _, _ = region
                    n_spans += 1
                    measurements: list[dict[str, object]] = []
                    rec["measurements"] = measurements
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
                        p_a = _next_dist(prefix)
                        p_c = _next_dist(replay)
                        kl = float(
                            (
                                p_a
                                * (
                                    p_a.clamp_min(1e-12).log()
                                    - p_c.clamp_min(1e-12).log()
                                )
                            ).sum()
                        )
                        flip = int(p_a.argmax()) != int(p_c.argmax())
                        by_dist[d].append((kl, flip))
                        measurements.append({"distance": d, "kl": kl, "flip": flip})

    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_out.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} generation records to {jsonl_out}")
    _report(model_name, n_spans, by_dist)


def verify_records(paths: list[str]) -> None:
    """CPU re-derivation of the decay table from saved records.

    Canonicality, the span region, and which prefixes still differ from their
    replay are recomputed from the raw token IDs; KL/flip values are read from
    the records (recomputing them needs the model).
    """
    from transformers import AutoTokenizer

    from common.artifacts import resolve_record_path
    from retok.phase2_probe import decode_for_roundtrip, roundtrip_is_measurable

    for pathlike in paths:
        path = Path(resolve_record_path(pathlike))
        records = [json.loads(x) for x in path.read_text().splitlines() if x]
        if not records:
            print(f"{path}: EMPTY")
            continue
        model_name = records[0]["model"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError as e:
            print(f"\n===== {model_name} =====")
            print(f"  SKIPPED — could not load tokenizer: {type(e).__name__}")
            print("  (gated repo? accept the license and `huggingface-cli login`)")
            continue
        by_dist: dict[int, list[tuple[float, bool]]] = {d: [] for d in DISTANCES}
        n_spans = 0
        mismatches = 0
        for rec in records:
            gen = rec["generated_ids"]
            text = decode_for_roundtrip(tokenizer, gen)
            measurable = roundtrip_is_measurable(tokenizer, text)
            if measurable != (not rec["excluded"]):
                mismatches += 1
                continue
            if not measurable:
                continue
            canon = tokenizer.encode(text, add_special_tokens=False)
            region = _first_replace(gen, canon)
            if (region is not None) != rec["non_canonical"]:
                mismatches += 1
                continue
            if region is None:
                continue
            n_spans += 1
            _, i2, _, _ = region
            stored = {m["distance"]: m for m in rec["measurements"]}
            for d in DISTANCES:
                end = i2 + d
                if end >= len(gen):
                    break
                prefix = gen[:end]
                ptext = decode_for_roundtrip(tokenizer, prefix)
                if not roundtrip_is_measurable(tokenizer, ptext):
                    continue
                replay = tokenizer.encode(ptext, add_special_tokens=False)
                measured = replay != prefix
                if measured != (d in stored):
                    mismatches += 1
                    continue
                if measured:
                    m = stored[d]
                    by_dist[d].append((m["kl"], bool(m["flip"])))
        _report(model_name, n_spans, by_dist)
        status = (
            "OK — recomputed canonicality/regions match the stored records"
            if mismatches == 0
            else f"MISMATCH — {mismatches} records disagree with recomputation"
        )
        print(f"\n  {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal contamination decay")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jsonl-out", help="Write per-generation records (with per-distance KL/flip)"
    )
    parser.add_argument(
        "--from-jsonl",
        nargs="+",
        metavar="PATH",
        help="No GPU: re-derive the table from saved records. hf:<name> reads "
        "the published dataset.",
    )
    args = parser.parse_args()
    if args.from_jsonl:
        verify_records(args.from_jsonl)
        return
    run(
        args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        jsonl_out=Path(args.jsonl_out) if args.jsonl_out else None,
    )


if __name__ == "__main__":
    main()
