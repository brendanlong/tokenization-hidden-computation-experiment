"""Non-canonical rate by the script the tokens are *actually* in — CPU only.

The per-domain rates elsewhere are keyed by the **prompt's** language. That is
not the same as the output's: asked a Russian prompt, Llama-3.2-1B answers
overwhelmingly in English, so its "multilingual_cyrillic" cell is measured on
text that is only ~11% Cyrillic. Llama-3.1-8B answers the same prompts ~82% in
Cyrillic. Comparing those two cells therefore compares different content, not
just different models.

This re-attributes every emitted token to the script of its own surface form and
recomputes the rate per script, which is comparable across models. Run:

    uv run python -m retok.phase2_script --all-published
    uv run python -m retok.phase2_script data/retok/artifacts/*.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from retok.phase2_probe import decode_for_roundtrip, roundtrip_is_measurable

# Scripts we have enough data to report. Everything else (punctuation, digits,
# whitespace, emoji) has no script and is counted only in the pooled total.
SCRIPTS = ("Latin", "Cyrillic", "CJK")
MIN_TOKENS = 300  # below this a cell is noise; report it as such rather than 0.0%


def token_script(surface: str) -> str | None:
    """Dominant script of a token's surface form, or None if it has no letters."""
    counts: dict[str, int] = defaultdict(int)
    for ch in surface:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if name.startswith("CYRILLIC"):
            counts["Cyrillic"] += 1
        elif name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            counts["CJK"] += 1
        elif name.startswith("LATIN"):
            counts["Latin"] += 1
    return max(counts, key=lambda k: counts[k]) if counts else None


def script_composition(text: str) -> dict[str, float]:
    """Fraction of the *letters* in ``text`` belonging to each script."""
    counts: dict[str, int] = defaultdict(int)
    for ch in text:
        s = token_script(ch)
        if s is not None:
            counts[s] += 1
    total = sum(counts.values())
    return {s: counts.get(s, 0) / total for s in SCRIPTS} if total else {}


def analyse(paths: list[Path]) -> None:
    print(f"{'model':<16}" + "".join(f"{s:>24}" for s in SCRIPTS))
    print(f"{'':<16}" + "".join(f"{'rate (tokens)':>24}" for _ in SCRIPTS))
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:
            continue
        model = rows[0]["model"]
        try:
            tok = AutoTokenizer.from_pretrained(model)
        except OSError:
            print(f"{model:<16}  SKIPPED (gated tokenizer)")
            continue
        agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for rec in rows:
            if rec["excluded"]:
                continue
            ids = rec["generated_ids"]
            text = decode_for_roundtrip(tok, ids)
            if not roundtrip_is_measurable(tok, text):
                continue
            canon = tok.encode(text, add_special_tokens=False)
            bad: set[int] = set()
            if canon != ids:
                for op, i1, i2, _, _ in difflib.SequenceMatcher(
                    a=ids, b=canon, autojunk=False
                ).get_opcodes():
                    if op != "equal":
                        bad.update(range(i1, i2))
            for i, t in enumerate(ids):
                s = token_script(tok.decode([t]))
                if s is None:
                    continue
                agg[s][1] += 1
                if i in bad:
                    agg[s][0] += 1
        cells = ""
        for s in SCRIPTS:
            n_bad, n_tok = agg[s]
            cell = f"{n_bad / n_tok:.2%} ({n_tok})" if n_tok >= MIN_TOKENS else "n<300"
            cells += f"{cell:>24}"
        print(f"{model.split('/')[-1][:15]:<16}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Local .jsonl paths")
    parser.add_argument(
        "--all-published",
        action="store_true",
        help="Analyse every per-model record file in the published dataset",
    )
    args = parser.parse_args()
    if args.all_published:
        # common.artifacts is the single source of truth for what is published
        # and where; phase2_probe keeps its own copy for the monorepo, which has
        # no HF resolver.
        from common.artifacts import artifact_path, published_record_files

        paths = [Path(artifact_path(name)) for name in published_record_files()]
    elif args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        parser.error("pass at least one path, or --all-published")
    analyse(paths)


if __name__ == "__main__":
    main()
