"""Phase-2 feasibility probe: does a real model generate non-canonically?

Detection is mechanical once you keep the actual token IDs the model emitted:
a generation is *non-canonical* iff re-encoding its own decoded text yields a
different token sequence — i.e. ``encode(decode(gen_ids)) != gen_ids``. That is
exactly what happens when a transcript is stored as text and re-tokenized: the
positions change. We measure the rate per domain and surface concrete spans,
prioritizing arithmetic (which mirrors the toy: digit-by-digit answer vs a
merged number token).

Run:
    uv run python -m retok.phase2_probe --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import difflib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.gpu import resolve_device

# The seven per-model record files published to the HF dataset, in the order
# they appear in the writeup's rate table.
PUBLISHED_RECORD_FILES = (
    "gpt2.jsonl",
    "meta-llama_Llama-3.2-1B-Instruct.jsonl",
    "Qwen_Qwen2.5-1.5B-Instruct.jsonl",
    "google_gemma-2-2b.jsonl",
    "meta-llama_Llama-3.2-3B-Instruct.jsonl",
    "meta-llama_Llama-3.1-8B-Instruct.jsonl",
    "openai_gpt-oss-20b.jsonl",
)

# Domain prompts. Arithmetic first (tightest match to the toy); then multilingual,
# code, and English prose as rate baselines.
PROMPTS: dict[str, list[str]] = {
    "arithmetic": [
        "Compute 48251 + 39987 and show your work digit by digit.",
        "What is 7823 * 941? Work through it step by step.",
        "Add these: 199999 + 800002 + 45678. Show each step.",
        "Multiply 1234 by 5678, showing the long multiplication.",
        "Sum the numbers 1 through 100, showing partial sums as you go.",
    ],
    # Latin-script languages are kept separate from CJK on purpose: in these
    # vocabularies CJK is ~1 character per token, so most CJK strings have only
    # ONE possible segmentation and structurally cannot be non-canonical. Mixing
    # them into one "multilingual" bucket makes a model tuned for CJK look
    # canonical for a reason that has nothing to do with the model.
    "multilingual_latin": [
        "Escribe un párrafo en español sobre la historia de Madrid.",
        "Écris un court poème en français sur la mer.",
        "Schreibe auf Deutsch über deutsche Küche und Bierkultur.",
        "Schreibe drei Sätze auf Deutsch über das Wetter heute.",
        "Escreva um parágrafo em português sobre o Rio de Janeiro.",
    ],
    "multilingual_cyrillic": [
        "Write three sentences in Russian about the weather today.",
        "Напиши абзац по-русски про русскую литературу.",
    ],
    "multilingual_cjk": [
        "日本語で東京の観光名所について3文で説明してください。",
        "用中文写一段关于北京的介绍。",
    ],
    "code": [
        "Write a Python function that computes the nth Fibonacci number.",
        "Write a bash one-liner to find the 10 largest files in a directory.",
        "Implement quicksort in Rust.",
        "Write a regex to validate an email address, with a short explanation.",
        "Implement a binary search tree in C++ with insert and search.",
        "Write a SQL query for the second-highest salary per department.",
        "Write a TypeScript function that debounces a callback.",
        "Write a Dockerfile for a Python Flask app.",
    ],
    "english": [
        "Explain how photosynthesis works.",
        "Tell a short story about a lighthouse keeper.",
        "Describe the plot of a mystery novel set on a train.",
    ],
}


@dataclass
class Span:
    """A maximal differing region between actual and canonical token streams."""

    surface: str
    actual_tokens: list[str]
    canonical_tokens: list[str]


def decode_for_roundtrip(tokenizer: object, ids: list[int]) -> str:
    """Decode with the destructive post-processing DISABLED.

    ``clean_up_tokenization_spaces`` ships **True** in Llama-3.x's
    ``tokenizer_config.json`` (and False for GPT-2/Qwen), and a per-model value
    overrides the library default. It applies ten destructive rewrites
    (``" ." -> "."``, ``" 's" -> "'s"``, …), which makes ``encode(decode(ids))``
    differ for reasons that have nothing to do with the model's segmentation —
    ~12% spurious non-canonical on ordinary English, **for Llama only**. Since
    that is exactly the direction of our cross-model comparison, it must be
    disabled explicitly rather than left to config defaults.
    """
    return tokenizer.decode(  # type: ignore[attr-defined]
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def roundtrip_is_measurable(tokenizer: object, text: str) -> bool:
    """False when the decode→encode round trip is unreliable for this text.

    Two known false-positive sources, both unrelated to the model's choices:

    - **U+FFFD**: byte-level decoding is lossy by construction, so a generation
      truncated mid-UTF-8-character can never re-encode. (Never fires on ASCII.)
    - **Unicode normalization**: Qwen2.5's tokenizer carries an NFC normalizer,
      so non-NFC text is silently normalized on encode and mismatches. This
      inflates *Qwen* specifically.
    """
    if "�" in text:
        return False
    return text == unicodedata.normalize("NFC", text)


def diff_spans(
    tokenizer: object, gen_ids: list[int]
) -> tuple[list[int], list[Span]] | None:
    """Return (canonical_ids, differing spans), or None if not measurable.

    A generation is non-canonical iff ``encode(decode(gen_ids)) != gen_ids``.
    Uses a proper sequence alignment so each returned span is a genuine
    replace-region (actual tokens vs the canonical tokens for the same
    substring), not a mis-trimmed slice. Returns ``None`` when the round trip
    is confounded (see :func:`roundtrip_is_measurable`) so callers can exclude
    the sample rather than count a false positive.
    """
    text = decode_for_roundtrip(tokenizer, gen_ids)
    if not roundtrip_is_measurable(tokenizer, text):
        return None
    canon = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
    if canon == gen_ids:
        return canon, []
    spans: list[Span] = []
    matcher = difflib.SequenceMatcher(a=gen_ids, b=canon, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        a_toks = [tokenizer.decode([t]) for t in gen_ids[i1:i2]]  # type: ignore[attr-defined]
        c_toks = [tokenizer.decode([t]) for t in canon[j1:j2]]  # type: ignore[attr-defined]
        spans.append(
            Span(surface="".join(a_toks), actual_tokens=a_toks, canonical_tokens=c_toks)
        )
    return canon, spans


@torch.no_grad()
def probe(
    model_name: str,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    dtype: str = "auto",
    jsonl_out: Path | None = None,
) -> None:
    device = resolve_device(require_cuda=False)
    print(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Default is "auto" = each checkpoint's AS-RELEASED precision, because we are
    # measuring what these models do as people actually run them. That means the
    # precision differs across models (GPT-2 and Gemma-2 ship fp32; Llama and
    # Qwen ship bf16) and is reported per model.
    #
    # This is not cosmetic: the metric is a tail-sampling measurement and is
    # precision-sensitive. Loading GPT-2 in bf16 instead of its native fp32 moved
    # its Cyrillic rate 0.43% -> 1.09%. Any rate quoted without the dtype is
    # therefore underspecified, exactly like the sampling parameters.
    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    # device_map streams shards straight to the GPU; without it from_pretrained
    # materializes the full model in HOST RAM first, which OOMs a 48 GB-RAM box
    # on a 20B model even though it fits in VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map={"": device.type},
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.manual_seed(seed)

    use_chat = tokenizer.chat_template is not None
    # domain -> dict of counters
    totals: dict[str, dict[str, int]] = {}
    examples: list[tuple[str, Span]] = []  # domain, differing span

    # Per-generation artifact records. This experiment argues that labs should
    # retain the token IDs a model actually emitted; it would be incoherent not
    # to do so ourselves. Every rate in the paper is recomputable from this file
    # with no GPU.
    records: list[dict[str, object]] = []

    n_excluded = 0
    for domain, prompts in PROMPTS.items():
        n_noncanon = 0
        n_total = 0
        noncanon_tok = 0  # actual tokens inside a differing region
        total_tok = 0
        n_spans = 0
        for prompt in prompts:
            if use_chat:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt
            enc = tokenizer(text, return_tensors="pt").to(device)
            prompt_len = enc.input_ids.shape[1]
            # Sampling params must be pinned EXPLICITLY: HF inherits anything
            # unset from the model repo's generation_config.json, and those
            # differ per model (Qwen2.5 ships top_k=20, top_p=0.8,
            # repetition_penalty=1.1; Llama-3.2 ships top_p=0.9; GPT-2 ships
            # nothing). Off-canonical tokens live in the tail, so inherited
            # truncation suppresses this metric by ~3x — a first-order
            # cross-model confound.
            # temperature=0 means greedy, which is deterministic: sampling
            # params are meaningless and repeated draws are identical, so the
            # only way to add evidence there is more prompts/models, not
            # n_samples. (Greedy is NOT non-canonical-free — see RESULTS.md.)
            if temperature == 0.0:
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
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
                gen_ids = seq[prompt_len:].tolist()
                # strip trailing pad/eos
                while gen_ids and gen_ids[-1] in (
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                ):
                    gen_ids.pop()
                if not gen_ids:
                    continue
                result = diff_spans(tokenizer, gen_ids)
                gen_text = decode_for_roundtrip(tokenizer, gen_ids)
                if result is None:
                    n_excluded += 1  # confounded round trip; not counted either way
                    records.append(
                        {
                            "model": model_name,
                            "domain": domain,
                            "prompt": prompt,
                            "seed": seed,
                            "temperature": temperature,
                            "excluded": True,
                            "exclude_reason": (
                                "replacement_char"
                                if "\ufffd" in gen_text
                                else "non_nfc"
                            ),
                            "generated_ids": gen_ids,
                            "text": gen_text,
                        }
                    )
                    continue
                canon_ids, spans = result
                n_total += 1
                total_tok += len(gen_ids)
                records.append(
                    {
                        "model": model_name,
                        "domain": domain,
                        "prompt": prompt,
                        "seed": seed,
                        "temperature": temperature,
                        "excluded": False,
                        "non_canonical": bool(spans),
                        # the whole point: what the model ACTUALLY emitted...
                        "generated_ids": gen_ids,
                        # ...and what storing-as-text-then-re-encoding gives you
                        "canonical_ids": canon_ids,
                        "text": gen_text,
                        "spans": [
                            {
                                "surface": sp.surface,
                                "actual": sp.actual_tokens,
                                "canonical": sp.canonical_tokens,
                            }
                            for sp in spans
                        ],
                    }
                )
                if spans:
                    n_noncanon += 1
                    n_spans += len(spans)
                    noncanon_tok += sum(len(s.actual_tokens) for s in spans)
                    if len(examples) < 40:
                        examples.append((domain, spans[0]))
        totals[domain] = {
            "n_noncanon": n_noncanon,
            "n_total": n_total,
            "noncanon_tok": noncanon_tok,
            "total_tok": total_tok,
            "n_spans": n_spans,
        }

    import transformers

    print("\n===== NON-CANONICAL GENERATION RATE =====")
    print(
        f"model={model_name}  temp={temperature}  "
        f"transformers={transformers.__version__}"
    )
    print(
        f"controls: dtype={dtype} (pinned), "
        "clean_up_tokenization_spaces=False, add_special_tokens=False, "
        "top_k=0/top_p=1.0/rep_penalty=1.0 (pinned, not inherited); "
        f"excluded {n_excluded} confounded round trips (U+FFFD / non-NFC)"
    )
    hdr = (
        f"{'domain':<14}{'gens w/ span':>13}{'per-gen':>9}"
        f"{'per-token':>11}{'spans/gen':>11}"
    )
    print(hdr)
    for domain, t in totals.items():
        per_gen = t["n_noncanon"] / max(1, t["n_total"])
        per_tok = t["noncanon_tok"] / max(1, t["total_tok"])
        spg = t["n_spans"] / max(1, t["n_total"])
        print(
            f"{domain:<14}{t['n_noncanon']:>6}/{t['n_total']:<6}{per_gen:>8.0%}"
            f"{per_tok:>10.2%}{spg:>11.2f}"
        )

    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_out.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"\nWrote {len(records)} generation records (with actual + canonical "
            f"token IDs) to {jsonl_out}"
        )

    print("\n===== EXAMPLE NON-CANONICAL SPANS (actual | canonical) =====")
    for domain, span in examples[:25]:
        actual = " ".join(repr(t) for t in span.actual_tokens)
        canonical = " ".join(repr(t) for t in span.canonical_tokens)
        print(
            f"[{domain}] surface={span.surface!r}\n"
            f"    actual   ({len(span.actual_tokens)} tok): {actual}\n"
            f"    canonical({len(span.canonical_tokens)} tok): {canonical}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-canonical generation probe")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="auto",
        help=(
            "Numerical precision. Default 'auto' = the checkpoint's as-released "
            "dtype, i.e. how the model is normally run. This metric is "
            "precision-sensitive, so the value used is reported with every run."
        ),
    )
    parser.add_argument(
        "--jsonl-out",
        default=None,
        help=(
            "Write one JSON record per generation, including the actual and "
            "canonical token IDs. Every reported rate is recomputable from this "
            "file without a GPU."
        ),
    )
    args = parser.parse_args()
    probe(
        args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        dtype=args.dtype,
        jsonl_out=Path(args.jsonl_out) if args.jsonl_out else None,
    )


if __name__ == "__main__":
    main()
