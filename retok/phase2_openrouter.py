"""Measure open-weight frontier models via OpenRouter logprobs.

Same round-trip check as phase2_api, generalised to models whose tokenizer
lives on HuggingFace rather than tiktoken. Comparison is over token BOUNDARIES
(byte offsets), which sidesteps string->id ambiguity entirely: the sampled
boundaries come from each logprob entry's exact bytes, the canonical boundaries
from re-encoding the decoded text.

What an afternoon of probing established about OpenRouter (2026-08-20):

- `require_parameters: true` does NOT guarantee logprobs actually come back;
  several providers claim support and return an empty list.
- Reasoning models (gpt-oss, R1, Qwen-thinking) return their traces in a
  separate `reasoning` field with NO logprobs, so **reasoning traces are not
  measurable through this API at all** — the blind spot from the closed models
  extends to open ones served via routers.
- Working with bytes as of today: DeepSeek V3-0324 / V3.1, Qwen3-235B
  (final content only). Llama endpoints are flaky and their tokenizers are
  gated anyway. Kimi and 405B expose no logprobs endpoints.
- Providers serve various quantizations (fp4/fp8/bf16). The serving provider is
  recorded per generation; rates at different quants are not strictly
  comparable (dtype moves tail sampling).

Asymmetry, as with the OpenAI path: a NONZERO rate is hard to fake, a ZERO
rate is ambiguous with provider-side re-serialization.

Run:
    OPEN_ROUTER_API_KEY=... uv run python -m retok.phase2_openrouter \\
        --models deepseek/deepseek-chat-v3.1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from retok.phase2_probe import PROMPTS

# OpenRouter slug -> HF tokenizer repo. Only ungated tokenizers are usable.
TOKENIZERS = {
    "deepseek/deepseek-chat-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "deepseek/deepseek-chat-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "qwen/qwen3-235b-a22b-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}


def _chat(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    providers: list[str] | None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 1,
        "provider": (
            {"order": providers, "allow_fallbacks": False}
            if providers
            else {"require_parameters": True}
        ),
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPEN_ROUTER_API_KEY']}",
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


def _boundaries(chunks: list[bytes]) -> list[tuple[int, int]]:
    out, pos = [], 0
    for c in chunks:
        out.append((pos, pos + len(c)))
        pos += len(c)
    return out


def measure(
    model: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    jsonl_out: Path | None,
    providers_pin: list[str] | None,
    top_p: float | None = None,
    top_k: int | None = None,
    prompt_set: str = "default",
    workers: int = 6,
) -> None:
    if prompt_set == "english-v2":
        from retok.phase2_english import ENGLISH_PROMPTS

        prompt_dict = ENGLISH_PROMPTS
    else:
        prompt_dict = PROMPTS
    tok = AutoTokenizer.from_pretrained(TOKENIZERS[model])
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    fidelity_failures: dict[str, int] = defaultdict(int)
    providers: dict[str, int] = defaultdict(int)
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

    tasks = [
        (domain, prompt)
        for domain, prompts in prompt_dict.items()
        for prompt in prompts
        for _ in range(n_samples)  # n>1 unsupported by some providers
    ]

    def safe_chat(task: tuple[str, str]) -> dict:
        try:
            return _chat(
                model, task[1], temperature, max_tokens, providers_pin, top_p, top_k
            )
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            return {"error": {"message": f"request-failed: {e}"}}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for (domain, prompt), resp in zip(
            tasks, pool.map(safe_chat, tasks), strict=True
        ):
            providers[resp.get("provider", "?")] += 1
            if not resp.get("choices"):
                # error payload (rate limit, provider fault) delivered
                # with HTTP 200 — count and move on
                err = str(resp.get("error", {}).get("message", "?"))[:60]
                fidelity_failures[f"no-choices:{err}"] += 1
                continue
            choice = resp["choices"][0]
            content = choice["message"]["content"] or ""
            lp = (choice.get("logprobs") or {}).get("content") or []
            prov = resp.get("provider", "?")
            if not lp or not lp[0].get("bytes"):
                fidelity_failures[f"{prov}:no-logprob-bytes"] += 1
                continue
            sampled = [bytes(t["bytes"]) for t in lp]
            if b"".join(sampled).decode("utf-8", "replace") != content:
                fidelity_failures[f"{prov}:bytes-content-mismatch"] += 1
                continue
            canon_ids = tok.encode(content, add_special_tokens=False)
            canon = [
                tok.decode([i], clean_up_tokenization_spaces=False).encode()
                for i in canon_ids
            ]
            if b"".join(canon) != b"".join(sampled):
                # tokenizer round trip altered the text (normalisation);
                # not measurable, same exclusion rule as phase2_probe
                fidelity_failures[f"{prov}:normalisation"] += 1
                continue
            sb, cb = _boundaries(sampled), _boundaries(canon)
            bad = 0
            if sb != cb:
                for op, i1, i2, _, _ in difflib.SequenceMatcher(
                    a=sb, b=cb, autojunk=False
                ).get_opcodes():
                    if op != "equal":
                        bad += i2 - i1
                if len(examples) < 5:
                    m = next(
                        o
                        for o in difflib.SequenceMatcher(
                            a=sb, b=cb, autojunk=False
                        ).get_opcodes()
                        if o[0] != "equal"
                    )
                    _, i1, i2, j1, j2 = m
                    a_s = [sampled[k].decode("utf-8", "replace") for k in range(i1, i2)]
                    c_s = [canon[k].decode("utf-8", "replace") for k in range(j1, j2)]
                    examples.append(f"[{domain}] {a_s} | canonical {c_s}")
            a = agg[domain]
            a[0] += int(sb != cb)
            a[1] += 1
            a[2] += bad
            a[3] += len(sampled)
            emit(
                {
                    "model": model,
                    "provider": resp.get("provider"),
                    "domain": domain,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "prompt_set": prompt_set,
                    "top_k": top_k,
                    "sampled_tokens": [s.decode("utf-8", "replace") for s in sampled],
                    "canonical_ids": canon_ids,
                    "text": content,
                    "non_canonical": sb != cb,
                    "excluded": False,
                }
            )

    if sink:
        sink.close()
    print(f"\n===== OPENROUTER: {model} (temp={temperature}) =====", flush=True)
    print(f"providers: {dict(providers)}")
    print(f"exclusions by provider/cause: {dict(fidelity_failures)}")
    print(f"{'domain':<24}{'non-canon':>12}{'per-gen':>9}{'per-token':>11}")
    for domain in sorted(agg):
        nc, n, bad, tot = agg[domain]
        print(
            f"{domain:<24}{nc:>5}/{n:<6}"
            f"{nc / max(1, n):>8.0%}{bad / max(1, tot):>10.2%}"
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
    p.add_argument("--models", nargs="+", default=["deepseek/deepseek-chat-v3.1"])
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--out-dir", default="data/retok/api")
    p.add_argument("--prompt-set", choices=["default", "english-v2"], default="default")
    p.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Pin serving providers (skip ones that fail fidelity checks)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="send top_p explicitly (omitted from the request when unset)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="send top_k explicitly (0 disables truncation; omitted when unset)",
    )
    a = p.parse_args()
    for model in a.models:
        safe = model.replace("/", "_")
        measure(
            model,
            a.n_samples,
            a.temperature,
            a.max_tokens,
            Path(a.out_dir) / f"{safe}.jsonl",
            a.providers,
            top_p=a.top_p,
            top_k=a.top_k,
            prompt_set=a.prompt_set,
            workers=a.workers,
        )


if __name__ == "__main__":
    main()
