"""Join english-v2 records with judge verdicts; report raw vs compliant rates.

    uv run python -m retok.phase2_english_report data/retok/english_v2

Expects ``<model>.jsonl`` + ``judge_<model>.jsonl`` pairs in the directory.
Per model: per-token non-canonical rate over all measurable generations
(raw), over compliant ones only (followed == "full" and coherent), the
compliance mix, and a per-register breakdown of the compliant rate. CPU-only.
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import defaultdict
from pathlib import Path


def tok_rate(rows: list[dict]) -> tuple[int, int]:
    bad = tot = 0
    for r in rows:
        ids, canon = r["generated_ids"], r["canonical_ids"]
        tot += len(ids)
        if r["non_canonical"]:
            for op, i1, i2, _, _ in difflib.SequenceMatcher(
                a=ids, b=canon, autojunk=False
            ).get_opcodes():
                if op != "equal":
                    bad += i2 - i1
    return bad, tot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    args = p.parse_args()
    d = Path(args.directory)

    for rec_path in sorted(d.glob("*.jsonl")):
        if rec_path.name.startswith("judge_"):
            continue
        judge_path = d / f"judge_{rec_path.name}"
        with rec_path.open() as fh:
            rows = [json.loads(line) for line in fh]
        if rows and "generated_ids" not in rows[0]:
            print(
                f"{rec_path.stem}: skipped (no generated_ids — OpenRouter "
                "boundary records; analyse via phase2_openrouter conventions)"
            )
            continue
        verdicts: dict[int, dict] = {}
        if judge_path.exists():
            for j in map(json.loads, judge_path.open()):
                verdicts[j["idx"]] = j

        measurable = [(i, r) for i, r in enumerate(rows) if not r.get("excluded")]
        raw_bad, raw_tot = tok_rate([r for _, r in measurable])
        name = rec_path.stem
        line = f"{name}: raw {raw_bad}/{raw_tot} = {raw_bad / max(1, raw_tot):.3%}"

        if verdicts:
            mix: dict[str, int] = defaultdict(int)
            compliant = []
            for i, r in measurable:
                v = verdicts.get(i, {})
                if "followed" not in v or v.get("domain") != r["domain"]:
                    # missing/errored/skipped verdict, or a stale sidecar
                    # whose rows no longer line up: ungraded, never compliant
                    mix["ungraded"] += 1
                    continue
                english = v.get("language", "").lower().startswith("english")
                key = v["followed"] + ("" if v["coherent"] else "/incoherent")
                if v["followed"] == "full" and v["coherent"] and not english:
                    key = "full/non-english"
                mix[key] += 1
                if v["followed"] == "full" and v["coherent"] and english:
                    compliant.append(r)
            c_bad, c_tot = tok_rate(compliant)
            line += (
                f" | compliant {c_bad}/{c_tot} = {c_bad / max(1, c_tot):.3%}"
                f" (n={len(compliant)}/{len(measurable)}) | mix {dict(mix)}"
            )
        print(line)

        if verdicts:
            by_reg: dict[str, list[dict]] = defaultdict(list)
            for i, r in measurable:
                v = verdicts.get(i, {})
                if (
                    v.get("followed") == "full"
                    and v.get("coherent")
                    and v.get("language", "").lower().startswith("english")
                    and v.get("domain") == r["domain"]
                ):
                    by_reg[r["domain"]].append(r)
            for reg in sorted(by_reg):
                b, t = tok_rate(by_reg[reg])
                print(f"    {reg:12s} compliant {b}/{t} = {b / max(1, t):.3%}")


if __name__ == "__main__":
    main()
