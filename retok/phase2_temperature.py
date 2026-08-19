"""Why is one model non-canonical and another not? Temperature vs immunity.

Two candidate explanations for a model generating ~0% non-canonically:

- **Structural**: its tokenizer offers no alternative segmentations. (Measurably
  false for Qwen2.5 vs Llama-3.2 — identical segmentation counts per word.)
- **Behavioral/entropy**: the model puts nearly all mass on the canonical
  continuation, so non-canonical tokens are only sampled from the tail.

The second predicts the rate is a **function of sampling temperature** and should
rise steeply with it; the first predicts ~0 at any temperature. This script
sweeps temperature per model and reports the non-canonical rate, which
distinguishes them.

Run (GPU):
    uv run python -m retok.phase2_temperature --jsonl-dir data/retok/artifacts

Re-derive the table from saved records (CPU only, recomputes canonicality from
the raw token IDs rather than trusting the stored flags; ``hf:`` paths read the
published dataset):

    uv run python -m retok.phase2_temperature \\
        --from-jsonl hf:temperature_meta-llama_Llama-3.2-1B-Instruct.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from retok.phase2_probe import PROMPTS, decode_for_roundtrip, diff_spans

# The fixed prompt set across temperatures (code+multilingual: where spans live).
SWEEP_PROMPTS = ("code", 4), ("multilingual_latin", 4)


def _sweep_prompts() -> list[tuple[str, str]]:
    return [(d, p) for d, n in SWEEP_PROMPTS for p in PROMPTS[d][:n]]


def _summarise(model_name: str, records: list[dict[str, object]]) -> None:
    """Print the per-temperature table from generation records.

    Recomputes nothing model-side — canonicality is re-derived from the stored
    token IDs by the caller (``verify_records``) or was just measured live.
    """
    # temperature -> [n_noncanon, n_total, noncanon_tok, total_tok]
    agg: dict[float, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for rec in records:
        if rec["excluded"]:
            continue
        t = agg[float(rec["temperature"])]  # type: ignore[arg-type]
        t[1] += 1
        t[3] += len(rec["generated_ids"])  # type: ignore[arg-type]
        if rec["non_canonical"]:
            t[0] += 1
            t[2] += sum(len(s["actual"]) for s in rec["spans"])  # type: ignore[index]
    print(f"\n===== {model_name} =====")
    print(f"{'temp':>6}{'non-canon':>12}{'gens':>7}{'per-gen':>9}{'per-token':>11}")
    for temp in sorted(agg):
        n_noncanon, n_total, noncanon_tok, total_tok = agg[temp]
        per_gen = n_noncanon / max(1, n_total)
        per_tok = noncanon_tok / max(1, total_tok)
        print(
            f"{temp:>6.1f}{n_noncanon:>12}{n_total:>7}{per_gen:>8.0%}{per_tok:>10.2%}"
        )


def sweep(
    model_name: str,
    temperatures: list[float],
    *,
    n_samples: int,
    max_new_tokens: int,
    seed: int,
    jsonl_out: Path | None = None,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common.gpu import resolve_device

    device = resolve_device(require_cuda=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map={"": device.type}, low_cpu_mem_usage=True
    )
    model.eval()
    use_chat = tokenizer.chat_template is not None

    records: list[dict[str, object]] = []
    with torch.no_grad():
        for temp in temperatures:
            torch.manual_seed(seed)
            for domain, prompt in _sweep_prompts():
                text = (
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if use_chat
                    else prompt
                )
                enc = tokenizer(text, return_tensors="pt").to(device)
                plen = enc.input_ids.shape[1]
                greedy = temp <= 0.0
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=not greedy,
                    **(
                        {}
                        if greedy
                        else {
                            "temperature": temp,
                            "top_k": 0,
                            "top_p": 1.0,
                            "repetition_penalty": 1.0,
                        }
                    ),
                    num_return_sequences=1 if greedy else n_samples,
                    pad_token_id=tokenizer.eos_token_id,
                )
                for seq in out:
                    gen = seq[plen:].tolist()
                    while gen and gen[-1] in (
                        tokenizer.eos_token_id,
                        tokenizer.pad_token_id,
                    ):
                        gen.pop()
                    if not gen:
                        continue
                    rec: dict[str, object] = {
                        "model": model_name,
                        "domain": domain,
                        "prompt": prompt,
                        "seed": seed,
                        "temperature": temp,
                        "generated_ids": gen,
                    }
                    result = diff_spans(tokenizer, gen)
                    if result is None:
                        # confounded round trip (U+FFFD / non-NFC)
                        rec.update(excluded=True, non_canonical=False, spans=[])
                    else:
                        canon_ids, spans = result
                        rec.update(
                            excluded=False,
                            non_canonical=bool(spans),
                            canonical_ids=canon_ids,
                            text=decode_for_roundtrip(tokenizer, gen),
                            spans=[
                                {
                                    "surface": sp.surface,
                                    "actual": sp.actual_tokens,
                                    "canonical": sp.canonical_tokens,
                                }
                                for sp in spans
                            ],
                        )
                    records.append(rec)
    _summarise(model_name, records)
    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_out.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} generation records to {jsonl_out}")


def verify_records(paths: list[str]) -> None:
    """CPU re-derivation: recompute canonicality from the stored token IDs."""
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
        mismatches = 0
        for rec in records:
            ids = rec["generated_ids"]
            text = decode_for_roundtrip(tokenizer, ids)
            measurable = roundtrip_is_measurable(tokenizer, text)
            if measurable != (not rec["excluded"]):
                mismatches += 1
                continue
            if not measurable:
                continue
            canon = tokenizer.encode(text, add_special_tokens=False)
            if (canon != ids) != rec["non_canonical"]:
                mismatches += 1
        _summarise(model_name, records)
        status = (
            "OK — recomputed canonicality matches the stored flags exactly"
            if mismatches == 0
            else f"MISMATCH — {mismatches} records disagree with recomputation"
        )
        print(f"  {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature vs non-canonical rate")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        ],
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.0, 0.7, 1.0, 1.5, 2.0]
    )
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jsonl-dir",
        help="Write per-model generation records to "
        "<dir>/temperature_<model>.jsonl (the published-artifact format)",
    )
    parser.add_argument(
        "--from-jsonl",
        nargs="+",
        metavar="PATH",
        help="No GPU: re-derive the table from saved records, recomputing "
        "canonicality from the raw token IDs. hf:<name> reads the published "
        "dataset.",
    )
    args = parser.parse_args()
    if args.from_jsonl:
        verify_records(args.from_jsonl)
        return
    for model_name in args.models:
        jsonl_out = (
            Path(args.jsonl_dir) / f"temperature_{model_name.replace('/', '_')}.jsonl"
            if args.jsonl_dir
            else None
        )
        sweep(
            model_name,
            args.temperatures,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            jsonl_out=jsonl_out,
        )


if __name__ == "__main__":
    main()
