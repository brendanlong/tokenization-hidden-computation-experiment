"""Prompted induction of non-canonical tokens, on API-measurable models.

The local result (phase2_induce): asking Llama-3.2-1B to "write 'key'
immediately followed by 'board', with no separator" yields the non-canonical
['key','board'] where canonical is ['keyboard'] — 93% of compliant trials,
with matched controls (repeat the word / answer a question with the word) at
0%. Compliance was the small models' limiting factor.

Frontier models follow instructions far better, so this asks the sharper
question: when compliance is near-perfect, does the *tokenization* follow the
prompt's semantic decomposition, or does the model emit the canonical form?

Per model, a concat pair is only *informative* if the canonical tokenization of
the compound differs from the [A][B] boundary split — pairs where canonical
already splits at the seam are skipped (the vacuous-pair trap from the local
run). Controls run on the same targets.

Run:
    OPENAI_API_KEY=... OPEN_ROUTER_API_KEY=... \\
        uv run python -m retok.phase2_induce_api --models gpt-4.1
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from retok.phase2_induce import CONCAT_PAIRS, QUESTIONS

# model id -> (route, tokenizer spec)
MODELS: dict[str, tuple[str, str]] = {
    "gpt-4o": ("openai", "tiktoken"),
    "gpt-4o-mini": ("openai", "tiktoken"),
    "gpt-4.1": ("openai", "tiktoken"),
    "gpt-4.1-mini": ("openai", "tiktoken"),
    "deepseek/deepseek-chat-v3.1": ("openrouter", "deepseek-ai/DeepSeek-V3.1"),
    "qwen/qwen3-235b-a22b-2507": (
        "openrouter",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ),
}


def _post(url: str, key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
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


def _sample(model: str, prompt: str, temperature: float) -> list[bytes] | None:
    """One generation; returns the sampled tokens' exact bytes, or None."""
    route, _ = MODELS[model]
    if route == "openai":
        resp = _post(
            "https://api.openai.com/v1/chat/completions",
            os.environ["OPENAI_API_KEY"],
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_completion_tokens": 40,
                "logprobs": True,
            },
        )
    else:
        resp = _post(
            "https://openrouter.ai/api/v1/chat/completions",
            os.environ["OPEN_ROUTER_API_KEY"],
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 40,
                "logprobs": True,
                "top_logprobs": 1,
                "provider": {"order": ["Mara", "GMICloud", "Parasail", "AtlasCloud"]},
            },
        )
    choice = resp["choices"][0]
    lp = (choice.get("logprobs") or {}).get("content") or []
    if not lp or not lp[0].get("bytes"):
        return None
    toks = [bytes(t["bytes"]) for t in lp]
    content = choice["message"]["content"] or ""
    if b"".join(toks).decode("utf-8", "replace") != content:
        return None
    return toks


class _Tok:
    """Uniform encode/decode over tiktoken and HF tokenizers."""

    def __init__(self, spec: str, model: str) -> None:
        if spec == "tiktoken":
            import tiktoken

            self._enc = tiktoken.encoding_for_model(model)
            self._hf = None
        else:
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(spec)
            self._enc = None

    def encode_surfaces(self, text: str) -> list[str]:
        if self._enc is not None:
            return [self._enc.decode([i]) for i in self._enc.encode(text)]
        assert self._hf is not None
        ids = self._hf.encode(text, add_special_tokens=False)
        return [self._hf.decode([i], clean_up_tokenization_spaces=False) for i in ids]


def _informative(tok: _Tok, a: str, b: str) -> bool:
    """A pair is informative iff canonical(a+b) does NOT split at the seam."""
    surfaces = tok.encode_surfaces(a + b)
    pos, seam = 0, len(a)
    for s in surfaces:
        pos += len(s)
        if pos == seam:
            return False  # canonical already splits exactly at a|b
    return True


def _covering(tokens: list[bytes], text: str, target: str) -> list[str] | None:
    """Surfaces of the sampled tokens covering the first occurrence of target."""
    idx = text.find(target)
    if idx < 0:
        return None
    out, pos = [], 0
    for t in tokens:
        s = t.decode("utf-8", "replace")
        start, end = pos, pos + len(s)
        if end > idx and start < idx + len(target):
            out.append(s)
        pos = end
    return out


def run_model(model: str, n_samples: int, temperature: float, jsonl_out: Path) -> None:
    tok = _Tok(MODELS[model][1], model)
    pairs = [(a, b) for a, b in CONCAT_PAIRS if _informative(tok, a, b)]
    skipped = [(a, b) for a, b in CONCAT_PAIRS if (a, b) not in pairs]
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # ind, prod, trials
    records: list[dict[str, object]] = []
    examples: list[str] = []

    def trial(probe: str, target: str, prompt: str) -> None:
        toks = _sample(model, prompt, temperature)
        stats[probe][2] += 1
        if toks is None:
            return
        text = b"".join(toks).decode("utf-8", "replace")
        produced = target in text
        induced = False
        actual: list[str] | None = None
        if produced:
            stats[probe][1] += 1
            actual = _covering(toks, text, target)
            canonical = tok.encode_surfaces(target)
            # compare boundary structure within the covered span
            joined = "".join(actual or [])
            induced = (
                joined.strip() == target
                and [s for s in actual or []] != canonical
                and "".join(canonical) == target
            )
            if induced:
                stats[probe][0] += 1
                if len(examples) < 6:
                    examples.append(f"{probe}/{target}: {actual} | canon {canonical}")
        records.append(
            {
                "model": model,
                "probe": probe,
                "target": target,
                "prompt": prompt,
                "temperature": temperature,
                "produced": produced,
                "induced": induced,
                "actual_tokens": actual,
                "canonical_tokens": tok.encode_surfaces(target),
                "text": text,
            }
        )

    for a, b in pairs:
        target = a + b
        for _ in range(n_samples):
            trial(
                "concat",
                target,
                f'Write the word "{a}" immediately followed by the word "{b}", '
                f"with no space, hyphen, or any other separator between them. "
                f"Output only the resulting single string and nothing else.",
            )
        trial(
            "repeat",
            target,
            f'Repeat this word exactly, on its own, with nothing else: "{target}"',
        )
        if target in QUESTIONS:
            trial("question", target, QUESTIONS[target])

    print(f"\n===== INDUCTION (API): {model} =====")
    if skipped:
        print(f"skipped vacuous pairs (canonical already splits): {skipped}")
    for probe in ("concat", "repeat", "question"):
        ind, prod, trials = stats[probe]
        print(
            f"  {probe:<9} trials={trials:>3}  produced={prod:>3} "
            f"({prod / max(1, trials):>4.0%})  induced={ind}/{prod}"
            f" ({ind / max(1, prod):>4.0%} of produced)"
        )
    for e in examples:
        print(f"    {e}")
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to {jsonl_out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out-dir", default="data/retok/api")
    a = p.parse_args()
    for model in a.models:
        safe = model.replace("/", "_")
        run_model(
            model,
            a.n_samples,
            a.temperature,
            Path(a.out_dir) / f"induce_{safe}.jsonl",
        )


if __name__ == "__main__":
    main()
