"""Recompute the Phase-2 rates from saved token IDs — CPU only, no GPU.

Generation needs a GPU; *analysis* should not. The artifact JSONL written by
``phase2_probe --jsonl-out`` stores, per generation, the token IDs the model
actually emitted. This script re-derives canonicality from those IDs using only
the tokenizer, so every rate in the writeup is checkable without reproducing the
generation step — and without trusting the ``canonical_ids`` we cached.

It deliberately recomputes rather than reading the stored ``non_canonical`` flag,
so it audits the analysis, not just the bookkeeping.

Run against local artifacts, or straight from the published dataset:

    uv run python -m retok.phase2_verify data/retok/artifacts/*.jsonl
    uv run python -m retok.phase2_verify --all-published

Verifying a model's records loads that model's *tokenizer* (a few MB, not the
weights). The Llama and Gemma tokenizers are gated on HuggingFace, so those
files need an accepted license and `huggingface-cli login`; the rest need no
account, and gated models are skipped rather than aborting the run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from common.artifacts import artifact_path, published_record_files
from retok.phase2_probe import (
    decode_for_roundtrip,
    roundtrip_is_measurable,
)


def verify(paths: list[Path]) -> int:
    exit_code = 0
    skipped: list[str] = []
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not records:
            print(f"{path}: EMPTY")
            continue
        if "generated_ids" not in records[0]:
            print(f"\n=== {path.name} ===")
            if "sampled_tokens" in records[0]:
                # OpenRouter logprobs runs store token *strings*, not IDs;
                # their rates are replayed by `retok.phase2_overview`.
                print("  SKIPPED — OpenRouter schema (sampled_tokens strings);")
                print("  verify via `uv run python -m retok.phase2_overview`")
            else:
                # e.g. induce_*.jsonl (induction trials) swept in by a glob —
                # different schema, verified by its own tooling.
                print("  SKIPPED — no generated_ids field (not a rate-table artifact)")
            continue
        model = records[0]["model"]
        try:
            tok = AutoTokenizer.from_pretrained(model)
        except OSError as e:
            # The Llama and Gemma tokenizers are gated: without an accepted
            # license this is the expected failure. Skip rather than abort, so
            # the models a reader *can* check still get verified.
            print(f"\n=== {model}  ({path.name}) ===")
            print(f"  SKIPPED — could not load tokenizer: {type(e).__name__}")
            print("  (gated repo? accept the license and `huggingface-cli login`)")
            skipped.append(model)
            continue
        print(f"\n=== {model}  ({path.name}, {len(records)} records) ===")

        # domain -> [n_noncanon, n_total, noncanon_tok, total_tok]
        agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        mismatches = 0
        for rec in records:
            ids = rec["generated_ids"]
            text = decode_for_roundtrip(tok, ids)
            measurable = roundtrip_is_measurable(tok, text)
            if measurable != (not rec["excluded"]):
                mismatches += 1
            if not measurable:
                continue
            canon = tok.encode(text, add_special_tokens=False)
            non_canon = canon != ids
            if "non_canonical" in rec and non_canon != rec["non_canonical"]:
                mismatches += 1
            # recount affected tokens via the same alignment the probe uses
            n_bad = 0
            if non_canon:
                import difflib

                for op, i1, i2, _, _ in difflib.SequenceMatcher(
                    a=ids, b=canon, autojunk=False
                ).get_opcodes():
                    if op != "equal":
                        n_bad += i2 - i1
            a = agg[rec["domain"]]
            a[0] += int(non_canon)
            a[1] += 1
            a[2] += n_bad
            a[3] += len(ids)

        print(f"{'domain':<24}{'non-canon':>12}{'per-gen':>9}{'per-token':>11}")
        for domain in sorted(agg):
            nc, n, bad, tot = agg[domain]
            print(
                f"{domain:<24}{nc:>5}/{n:<6}{nc / max(1, n):>8.0%}"
                f"{bad / max(1, tot):>10.2%}"
            )
        if mismatches:
            print(f"  !! {mismatches} record(s) disagree with the stored flags")
            exit_code = 1
        else:
            print("  OK — recomputed canonicality matches the stored flags exactly")
    if skipped:
        print(f"\nSkipped {len(skipped)} model(s) whose tokenizer would not load:")
        for model in skipped:
            print(f"  - {model}")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase-2 rates from token IDs")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Local .jsonl paths, or hf:<name>.jsonl to pull from the dataset",
    )
    parser.add_argument(
        "--all-published",
        action="store_true",
        help="Verify every per-model record file in the published dataset",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.all_published:
        paths += [Path(artifact_path(name)) for name in published_record_files()]
    paths += [
        Path(artifact_path(p.removeprefix("hf:"))) if p.startswith("hf:") else Path(p)
        for p in args.paths
    ]
    if not paths:
        parser.error("pass at least one path, or --all-published")
    raise SystemExit(verify(paths))


if __name__ == "__main__":
    main()
