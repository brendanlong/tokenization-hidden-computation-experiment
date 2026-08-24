"""Task: decimal expansion of 1/b, and the divisor splits.

The numerator is fixed at 1, so this is 98-way sequence recall rather than also
requiring rotation-finding (the expansion of a/b is a rotation of 1/b's cycle).
Divisors are split train/held-out so we can report whether anything generalises,
without presuming it will not.
"""

from __future__ import annotations

from dataclasses import dataclass

# b=1 is excluded: 1/1 expands to all zeros, which is degenerate.
ALL_DIVISORS = tuple(range(2, 100))

# Terminating decimals: b whose only prime factors are 2 and 5, so the expansion
# is a short prefix followed by zeros. GPT-2 has a single 16-character token for
# sixteen zeros, so one correct prediction can buy 16 digits of reward here.


def _reduce25(b: int) -> int:
    while b % 2 == 0:
        b //= 2
    while b % 5 == 0:
        b //= 5
    return b


def cycle_length(b: int) -> int:
    """Length of the repeating cycle of 1/b (0 if it terminates)."""
    m = _reduce25(b)
    if m == 1:
        return 0
    r, n = 10 % m, 1
    while r != 1:
        r = (r * 10) % m
        n += 1
    return n


def divisor_class(b: int) -> str:
    """terminating / short-cycle / long-cycle — token economics differ hugely."""
    n = cycle_length(b)
    if n == 0:
        return "terminating"
    return "short-cycle" if n <= 6 else "long-cycle"


def expansion(b: int, places: int) -> str:
    """First ``places`` digits after the decimal point of 1/b."""
    r, out = 1 % b, []
    for _ in range(places):
        r *= 10
        out.append(str(r // b))
        r %= b
    return "".join(out)


def split_divisors(
    n_held_out: int, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministic train/held-out divisor split, stratified by class.

    Stratified so the held-out set is not accidentally all long-cycle, which
    would make the generalisation eval harder than the training distribution.
    """
    import random

    rng = random.Random(seed)
    by_class: dict[str, list[int]] = {}
    for b in ALL_DIVISORS:
        by_class.setdefault(divisor_class(b), []).append(b)
    held: list[int] = []
    for cls in sorted(by_class):
        pool = sorted(by_class[cls])
        rng.shuffle(pool)
        k = max(1, round(n_held_out * len(pool) / len(ALL_DIVISORS)))
        held.extend(pool[:k])
    held_set = set(held)
    train = tuple(b for b in ALL_DIVISORS if b not in held_set)
    return train, tuple(sorted(held_set))


@dataclass(frozen=True)
class Example:
    b: int
    prompt: str
    target: str


def build(divisors: tuple[int, ...], places: int) -> list[Example]:
    """One example per divisor. The prompt ends at '0.' with NO trailing space.

    Trailing whitespace is a real footgun here: GPT-2 encodes a number together
    with its preceding space, so a prompt ending in a bare space leaves a
    dangling token and makes the intended continuation unreachable. It silently
    invalidated three earlier runs in this line of work.
    """
    return [
        Example(b=b, prompt=f"1/{b} = 0.", target=expansion(b, places))
        for b in divisors
    ]
