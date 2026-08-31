"""Measure non-canonical output of CLOSED frontier models via API logprobs.

The chat completions API returns, per sampled token, its string and its exact
bytes. Concatenating the bytes reconstructs the output text; re-encoding that
text with the model's public tokenizer (tiktoken) gives the canonical
segmentation; comparing the two is the same round-trip check as everywhere else
in this experiment — no weights needed.

Two fidelity checks run per generation, because a provider could in principle
re-serialize tokens before returning them:

- the concatenated logprob bytes must equal the returned message content
  (proves we reconstructed the exact string the model produced);
- every returned token's bytes must be a real single token of the model's
  encoding (a re-serializer would tend to produce merge-order chunks, which
  this cannot distinguish from genuine canonical output — but non-o200k chunks
  would expose it outright).

The measurement is therefore asymmetric: a NONZERO rate is strong evidence
(hard to fake by accident), while a ZERO rate is ambiguous between "model is
canonical" and "provider canonicalises before returning". Note gpt-5 refuses
logprobs entirely ("You are not allowed to request logprobs from this model"),
so the most heavily RL-trained models cannot be measured this way at all —
worth stating in any writeup, since those are exactly the models the threat
model concerns.

Run:
    OPENAI_API_KEY=... uv run python -m retok.phase2_api \\
        --models gpt-4o gpt-4.1 --n-samples 4
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import tiktoken

from retok.phase2_probe import PROMPTS


def _chat(
    model: str,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    top_p: float | None = None,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "n": n,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "logprobs": True,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload).encode(),
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def measure(
    model: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    jsonl_out: Path | None,
    top_p: float | None = None,
    prompt_set: str = "default",
) -> None:
    if prompt_set == "english-v2":
        from retok.phase2_english import ENGLISH_PROMPTS

        prompt_dict = ENGLISH_PROMPTS
    else:
        prompt_dict = PROMPTS
    enc = tiktoken.encoding_for_model(model)
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    fidelity_failures = 0
    records: list[dict[str, object]] = []
    sink = None
    if jsonl_out:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        sink = jsonl_out.open("w")

    def emit(rec: dict) -> None:
        emit(rec)
        if sink:
            sink.write(json.dumps(rec) + "\n")
            sink.flush()

    examples: list[str] = []

    for domain, prompts in prompt_dict.items():
        for prompt in prompts:
            resp = _chat(model, prompt, n_samples, temperature, max_tokens, top_p)
            for choice in resp["choices"]:
                content = choice["message"]["content"] or ""
                lp = (choice.get("logprobs") or {}).get("content") or []
                if not lp:
                    continue
                # exact bytes of each sampled token
                tok_bytes = [bytes(t["bytes"]) for t in lp]
                # fidelity 1: bytes reconstruct the content exactly
                if b"".join(tok_bytes).decode("utf-8", "replace") != content:
                    fidelity_failures += 1
                    continue
                # fidelity 2: every token is a real single token of the encoding
                try:
                    ids = [enc.encode_single_token(b) for b in tok_bytes]
                except KeyError:
                    fidelity_failures += 1
                    continue
                canon = enc.encode(content)
                bad = 0
                if canon != ids:
                    for op, i1, i2, _, _ in difflib.SequenceMatcher(
                        a=ids, b=canon, autojunk=False
                    ).get_opcodes():
                        if op != "equal":
                            bad += i2 - i1
                    if len(examples) < 5:
                        m = next(
                            (
                                o
                                for o in difflib.SequenceMatcher(
                                    a=ids, b=canon, autojunk=False
                                ).get_opcodes()
                                if o[0] != "equal"
                            ),
                            None,
                        )
                        if m:
                            _, i1, i2, j1, j2 = m
                            actual = [enc.decode([t]) for t in ids[i1:i2]]
                            can = [enc.decode([t]) for t in canon[j1:j2]]
                            examples.append(f"[{domain}] {actual} | canonical {can}")
                a = agg[domain]
                a[0] += int(canon != ids)
                a[1] += 1
                a[2] += bad
                a[3] += len(ids)
                emit(
                    {
                        "model": model,
                        "domain": domain,
                        "prompt": prompt,
                        "temperature": temperature,
                        "top_p": top_p,
                        "prompt_set": prompt_set,
                        "generated_ids": ids,
                        "canonical_ids": canon,
                        "text": content,
                        "non_canonical": canon != ids,
                        "excluded": False,
                    }
                )

    if sink:
        sink.close()
    print(f"\n===== API MEASUREMENT: {model} (temp={temperature}) =====", flush=True)
    print(f"fidelity failures: {fidelity_failures} (bytes/token mismatches)")
    print(f"{'domain':<24}{'non-canon':>12}{'per-gen':>9}{'per-token':>11}")
    for domain in sorted(agg):
        nc, n, bad, tot = agg[domain]
        print(
            f"{domain:<24}{nc:>5}/{n:<6}{nc / max(1, n):>8.0%}"
            f"{bad / max(1, tot):>10.2%}"
        )
    tot_bad = sum(a[2] for a in agg.values())
    tot_tok = sum(a[3] for a in agg.values())
    print(f"{'POOLED':<24}{'':>12}{'':>9}{tot_bad / max(1, tot_tok):>10.2%}")
    for e in examples:
        print(f"    {e}")
    if jsonl_out:
        print(f"wrote {len(records)} records to {jsonl_out} (incrementally)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-4.1"])
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="send top_p explicitly (omitted from the request when unset)",
    )
    p.add_argument("--out-dir", default="data/retok/api")
    p.add_argument("--prompt-set", choices=["default", "english-v2"], default="default")
    a = p.parse_args()
    for model in a.models:
        measure(
            model,
            a.n_samples,
            a.temperature,
            a.max_tokens,
            Path(a.out_dir) / f"{model}.jsonl",
            top_p=a.top_p,
            prompt_set=a.prompt_set,
        )


if __name__ == "__main__":
    main()
