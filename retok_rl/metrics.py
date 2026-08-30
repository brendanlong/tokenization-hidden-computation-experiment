"""Tokenization metrics: which of the three attractors did the output land on?

There are two ways to be non-canonical and only one way to be canonical:

- **all-single-digit** — the prediction, and the easiest per-step choice
- **greedy-longest** — reward-dense chunks, what a *memorising* policy might
  prefer, since one correct prediction can cover 2-16 digits
- **canonical** — BPE's merge-order segmentation, which is neither the shortest
  nor the longest description

Greedy-longest differs from canonical for 93 of 98 divisors, so a null result
requires landing on precisely the canonical segmentation.
"""

from __future__ import annotations

from collections import Counter


def greedy_longest(tokenizer: object, digits: str, maxlen: int = 24) -> list[str]:
    """Segment ``digits`` by repeatedly taking the longest single-token prefix."""
    out: list[str] = []
    i = 0
    while i < len(digits):
        for length in range(min(maxlen, len(digits) - i), 0, -1):
            piece = digits[i : i + length]
            if len(tokenizer.encode(piece, add_special_tokens=False)) == 1:  # type: ignore[attr-defined]
                out.append(piece)
                i += length
                break
        else:
            out.append(digits[i])
            i += 1
    return out


def leading_correct(produced: str, target: str) -> int:
    """Number of correct leading digits — the reward, computed on TEXT."""
    n = 0
    for a, b in zip(produced, target, strict=False):
        if a != b:
            break
        n += 1
    return n


def digit_prefix(text: str) -> str:
    """Leading digit run of a completion, ignoring one optional leading space."""
    out = ""
    for ch in text.lstrip():
        if ch.isdigit():
            out += ch
        else:
            break
    return out


def classify(tokenizer: object, token_ids: list[int], digits: str) -> str:
    """Which attractor does this token sequence match, over the digit run?

    Compares the emitted tokens covering ``digits`` against the three reference
    segmentations. Anything else is "other" (a mixture).
    """
    emitted = [tokenizer.decode([t]) for t in token_ids]  # type: ignore[attr-defined]
    if "".join(emitted).strip() != digits:
        return "other"
    canon = [
        tokenizer.decode([t])  # type: ignore[attr-defined]
        for t in tokenizer.encode(digits, add_special_tokens=False)  # type: ignore[attr-defined]
    ]
    stripped = [e.strip() if i == 0 else e for i, e in enumerate(emitted)]
    if all(len(e) == 1 for e in stripped):
        return "all-single-digit"
    if stripped == canon:
        return "canonical"
    if stripped == greedy_longest(tokenizer, digits):
        return "greedy-longest"
    return "other"


def tokens_covering_digits(
    tokenizer: object, token_ids: list[int]
) -> tuple[list[int], str]:
    """The prefix of ``token_ids`` whose surface is the leading digit run."""
    kept: list[int] = []
    surface = ""
    for t in token_ids:
        piece = tokenizer.decode([t])  # type: ignore[attr-defined]
        probe = (surface + piece).lstrip()
        if probe and all(c.isdigit() for c in probe):
            kept.append(t)
            surface += piece
        else:
            break
    return kept, surface.lstrip()


def summarise(
    tokenizer: object, rollouts: list[tuple[list[int], str]]
) -> dict[str, float]:
    """Aggregate tokenization stats over (completion_ids, target) pairs.

    ``completion_ids`` may be the full emitted completion (specials and any
    non-digit tail included); the digit run is extracted here. Alongside the
    original attractor mix (which classifies the emitted digit run against
    the canonical/greedy/single segmentations of *its own* surface), this
    reports the corrected metric families added after Run 5:

    - **Round-trip canonicality** of the whole emitted stream
      (``encode(decode(ids)) != ids``), per generation and per token, with
      U+FFFD exclusions. An unexpected-but-canonical token is not a
      non-canonical token.
    - **Correct-region** metrics — per token, only over tokens lying
      entirely within the correct leading digits (matching the target), so
      wrong digits and trailing text are excluded by construction — plus a
      per-position single-digit breakdown (``pos_single_digit/{1..4}``).
    """
    import difflib

    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    attractors: Counter[str] = Counter()
    single_toks = total_toks = 0
    reward_sum = 0
    digits_sum = 0
    gen_nc = tok_bad = tok_total = excluded = 0
    run_nc = run_n = 0
    cr_single = cr_total = cr_nc = cr_n = 0
    pos_single: Counter[int] = Counter()
    pos_n: Counter[int] = Counter()
    for ids, target in rollouts:
        core = [t for t in ids if t not in special]
        kept, digits = tokens_covering_digits(tokenizer, core)
        n_correct = leading_correct(digits, target)
        reward_sum += n_correct
        digits_sum += len(digits)
        # round-trip canonicality of the full emitted stream
        if core:
            text = tokenizer.decode(core)  # type: ignore[attr-defined]
            if "�" in text:
                excluded += 1
            else:
                canon = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
                tok_total += len(core)
                if canon != core:
                    gen_nc += 1
                    for op, i1, i2, _, _ in difflib.SequenceMatcher(
                        a=core, b=canon, autojunk=False
                    ).get_opcodes():
                        if op != "equal":
                            tok_bad += i2 - i1
        # round-trip canonicality of the digit run alone
        if kept:
            run_n += 1
            canon_run = tokenizer.encode(digits, add_special_tokens=False)  # type: ignore[attr-defined]
            canon_run_sp = tokenizer.encode(  # type: ignore[attr-defined]
                " " + digits, add_special_tokens=False
            )
            run_nc += kept != canon_run and kept != canon_run_sp
        # correct-region: tokens entirely within the correct leading digits
        if n_correct and kept:
            cr_ids: list[int] = []
            cr_surface = ""
            consumed = 0
            for i, t in enumerate(kept):
                piece = tokenizer.decode([t])  # type: ignore[attr-defined]
                stripped = piece.lstrip() if i == 0 else piece
                if consumed + len(stripped) > n_correct:
                    break  # straddles the correct/incorrect boundary
                for j in range(consumed + 1, consumed + len(stripped) + 1):
                    pos_n[j] += 1
                    pos_single[j] += len(stripped) == 1
                cr_ids.append(t)
                cr_surface += stripped
                cr_total += 1
                cr_single += len(stripped) == 1
                consumed += len(stripped)
            if cr_ids:
                cr_n += 1
                canon_cr = tokenizer.encode(cr_surface, add_special_tokens=False)  # type: ignore[attr-defined]
                canon_cr_sp = tokenizer.encode(  # type: ignore[attr-defined]
                    " " + cr_surface, add_special_tokens=False
                )
                cr_nc += cr_ids != canon_cr and cr_ids != canon_cr_sp
        if not kept:
            attractors["empty"] += 1
            continue
        attractors[classify(tokenizer, kept, digits)] += 1
        for i, t in enumerate(kept):
            piece = tokenizer.decode([t])  # type: ignore[attr-defined]
            piece = piece.strip() if i == 0 else piece
            total_toks += 1
            single_toks += len(piece) == 1
    n = max(1, len(rollouts))
    n_meas = max(1, len(rollouts) - excluded)
    out: dict[str, float] = {
        "mean_reward": reward_sum / n,
        "mean_digits_emitted": digits_sum / n,
        "frac_single_digit_tokens": single_toks / max(1, total_toks),
        "roundtrip/gen_non_canonical": gen_nc / n_meas,
        "roundtrip/tok_non_canonical": tok_bad / max(1, tok_total),
        "roundtrip/digitrun_non_canonical": run_nc / max(1, run_n),
        "roundtrip/excluded": excluded / n,
        "correct_region/frac_single_digit": cr_single / max(1, cr_total),
        "correct_region/non_canonical": cr_nc / max(1, cr_n),
        "correct_region/mean_tokens": cr_total / max(1, cr_n),
    }
    for name in ("all-single-digit", "canonical", "greedy-longest", "other", "empty"):
        out[f"attractor/{name}"] = attractors[name] / n
    for j in range(1, 5):
        out[f"pos_single_digit/{j}"] = pos_single[j] / max(1, pos_n[j])
        out[f"pos_n/{j}"] = float(pos_n[j])
    return out
