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
    """Aggregate tokenization stats over (token_ids, target) pairs."""
    attractors: Counter[str] = Counter()
    single_toks = total_toks = 0
    reward_sum = 0
    digits_sum = 0
    for ids, target in rollouts:
        kept, digits = tokens_covering_digits(tokenizer, ids)
        reward_sum += leading_correct(digits, target)
        digits_sum += len(digits)
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
    out: dict[str, float] = {
        "mean_reward": reward_sum / n,
        "mean_digits_emitted": digits_sum / n,
        "frac_single_digit_tokens": single_toks / max(1, total_toks),
    }
    for name in ("all-single-digit", "canonical", "greedy-longest", "other", "empty"):
        out[f"attractor/{name}"] = attractors[name] / n
    return out
