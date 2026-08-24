"""One consistent rate table for every model measured, local and API — CPU only.

Recomputes the pooled per-token non-canonical rate from per-generation records
for all 17 models, with one method (aligned diff, ``autojunk=False``) and one
prompt distribution (the 25-prompt set). This is the single source of truth for
the overview figure: `figures.py` embeds these values and cites this module,
so anyone can check them by running:

    uv run python -m retok.phase2_overview

Record sources, in the published dataset (`--data-dir` overrides with a local
directory of the same layout):

    <root>/<model>.jsonl              local-weights runs (original 7)
    lw_comparison/<model>.jsonl       Gemma-1/Gemma-3/Llama-2 comparison runs
    api/<model>.jsonl                 API-logprobs runs (OpenAI + OpenRouter)

Gated tokenizers (Llama, Gemma) are skipped with a message rather than
aborting, as in `phase2_verify`. A model with 0 observed non-canonical tokens
is reported with its detection floor (1/total_tokens): "0 observed in N tokens"
is not "rate 0".
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

DATASET = "brendanlong/retok-noncanonical-tokenization"

# (display name, dataset-relative path, kind)
#   hf      — generated_ids + a HF tokenizer looked up from the record's model id
#   openai  — generated_ids + canonical_ids stored; tiktoken for surfaces
#   router  — sampled token strings + canonical_ids; HF tokenizer for surfaces
MODELS: list[tuple[str, str, str]] = [
    ("GPT-2", "gpt2.jsonl", "hf"),
    ("Llama-3.2-1B", "meta-llama_Llama-3.2-1B-Instruct.jsonl", "hf"),
    ("Qwen2.5-1.5B", "Qwen_Qwen2.5-1.5B-Instruct.jsonl", "hf"),
    ("Gemma-2-2b", "google_gemma-2-2b.jsonl", "hf"),
    ("Llama-3.2-3B", "meta-llama_Llama-3.2-3B-Instruct.jsonl", "hf"),
    ("Llama-3.1-8B", "meta-llama_Llama-3.1-8B-Instruct.jsonl", "hf"),
    ("gpt-oss-20b", "openai_gpt-oss-20b.jsonl", "hf"),
    ("Gemma-1-2B", "lw_comparison/google_gemma-2b-it.jsonl", "hf"),
    ("Gemma-3-4B", "lw_comparison/google_gemma-3-4b-it.jsonl", "hf"),
    ("Llama-2-7B", "lw_comparison/meta-llama_Llama-2-7b-chat-hf.jsonl", "hf"),
    ("gpt-4o", "api/gpt-4o.jsonl", "openai"),
    ("gpt-4o-mini", "api/gpt-4o-mini.jsonl", "openai"),
    ("gpt-4.1", "api/gpt-4.1.jsonl", "openai"),
    ("gpt-4.1-mini", "api/gpt-4.1-mini.jsonl", "openai"),
    ("Qwen3-235B", "api/qwen_qwen3-235b-a22b-2507.jsonl", "router"),
    ("DeepSeek-V3", "api/deepseek_deepseek-chat-v3-0324.jsonl", "router"),
    ("DeepSeek-V3.1", "api/deepseek_deepseek-chat-v3.1.jsonl", "router"),
]


def _bad_tokens(a: list, b: list) -> int:
    if a == b:
        return 0
    return sum(
        i2 - i1
        for op, i1, i2, _, _ in difflib.SequenceMatcher(
            a=a, b=b, autojunk=False
        ).get_opcodes()
        if op != "equal"
    )


def _bounds(chunks: list[str]) -> list[tuple[int, int]]:
    out, pos = [], 0
    for c in chunks:
        out.append((pos, pos + len(c)))
        pos += len(c)
    return out


def rate(path: Path, kind: str) -> tuple[int, int]:
    """(non-canonical tokens, total tokens) for one record file."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    bad = tot = 0
    if kind == "hf":
        from transformers import AutoTokenizer

        from retok.phase2_probe import (
            decode_for_roundtrip,
            roundtrip_is_measurable,
        )

        tok = AutoTokenizer.from_pretrained(rows[0]["model"])
        for r in rows:
            if r["excluded"]:
                continue
            ids = r["generated_ids"]
            text = decode_for_roundtrip(tok, ids)
            if not roundtrip_is_measurable(tok, text):
                continue
            canon = tok.encode(text, add_special_tokens=False)
            tot += len(ids)
            bad += _bad_tokens(ids, canon)
    elif kind == "openai":
        for r in rows:
            tot += len(r["generated_ids"])
            bad += _bad_tokens(r["generated_ids"], r["canonical_ids"])
    else:  # router: compare byte boundaries, not ids
        from transformers import AutoTokenizer

        from retok.phase2_openrouter import TOKENIZERS

        tok = AutoTokenizer.from_pretrained(TOKENIZERS[rows[0]["model"]])
        for r in rows:
            sampled = r["sampled_tokens"]
            canon = [
                tok.decode([i], clean_up_tokenization_spaces=False)
                for i in r["canonical_ids"]
            ]
            tot += len(sampled)
            bad += _bad_tokens(_bounds(sampled), _bounds(canon))
    return bad, tot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default=None,
        help="Local directory mirroring the dataset layout; default downloads "
        "from the published dataset",
    )
    args = p.parse_args()

    print(f"{'model':<15}{'non-canon':>10}{'tokens':>9}{'per-token':>11}{'floor':>9}")
    for name, rel, kind in MODELS:
        if args.data_dir:
            path = Path(args.data_dir) / rel
        else:
            from huggingface_hub import hf_hub_download

            try:
                path = Path(hf_hub_download(DATASET, rel, repo_type="dataset"))
            except Exception as e:
                print(f"{name:<15}  UNPUBLISHED ({type(e).__name__})")
                continue
        try:
            bad, tot = rate(path, kind)
        except OSError:
            print(f"{name:<15}  SKIPPED (gated tokenizer)")
            continue
        print(
            f"{name:<15}{bad:>10}{tot:>9}{bad / max(1, tot):>10.3%}"
            f"{1 / max(1, tot):>9.3%}"
        )


if __name__ == "__main__":
    main()
