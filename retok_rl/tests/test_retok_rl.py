"""CPU tests for the task definition and the tokenization metrics."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from retok_rl.data import (
    ALL_DIVISORS,
    build,
    cycle_length,
    divisor_class,
    expansion,
    split_divisors,
)
from retok_rl.metrics import (
    classify,
    digit_prefix,
    greedy_longest,
    leading_correct,
    summarise,
    tokens_covering_digits,
)


def test_expansion_matches_long_division() -> None:
    assert expansion(7, 12) == "142857142857"
    assert expansion(2, 6) == "500000"
    assert expansion(3, 5) == "33333"


def test_cycle_length_and_class() -> None:
    assert cycle_length(2) == 0 and divisor_class(2) == "terminating"
    assert cycle_length(3) == 1 and divisor_class(3) == "short-cycle"
    assert cycle_length(7) == 6 and divisor_class(7) == "short-cycle"
    assert cycle_length(17) == 16 and divisor_class(17) == "long-cycle"


def test_divisor_split_is_disjoint_and_stratified() -> None:
    train, held = split_divisors(20, seed=0)
    assert not set(train) & set(held)
    assert set(train) | set(held) == set(ALL_DIVISORS)
    # every class represented in the held-out set, else the generalisation eval
    # is harder or easier than the training distribution
    assert {divisor_class(b) for b in held} == {
        "terminating",
        "short-cycle",
        "long-cycle",
    }


def test_prompt_has_no_trailing_space() -> None:
    """A trailing space makes the intended continuation unreachable in GPT-2.

    The tokenizer encodes a number together with its preceding space, so a
    prompt ending in ' ' leaves a dangling token. This silently invalidated
    three earlier runs; pin it.
    """
    for ex in build((7, 13), 10):
        assert not ex.prompt.endswith(" ")
        assert ex.prompt.endswith("0.")


def test_leading_correct_and_digit_prefix() -> None:
    assert leading_correct("1428", "142857") == 4
    assert leading_correct("1438", "142857") == 2
    assert leading_correct("", "142857") == 0
    assert digit_prefix("  142857 and then words") == "142857"
    assert digit_prefix("no digits") == ""


def test_reward_is_tokenization_agnostic() -> None:
    """The whole design: identical text scores identically however it is split."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    digits = expansion(7, 12)
    canon = tok.encode(digits, add_special_tokens=False)
    single = [tok.encode(c, add_special_tokens=False)[0] for c in digits]
    assert canon != single
    a, _ = tokens_covering_digits(tok, canon)
    b, _ = tokens_covering_digits(tok, single)
    assert leading_correct(tok.decode(a), digits) == leading_correct(
        tok.decode(b), digits
    )


def test_classify_distinguishes_three_attractors() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    digits = expansion(7, 12)
    canon = tok.encode(digits, add_special_tokens=False)
    single = [tok.encode(c, add_special_tokens=False)[0] for c in digits]
    greedy = [
        tok.encode(p, add_special_tokens=False)[0] for p in greedy_longest(tok, digits)
    ]
    assert classify(tok, canon, digits) == "canonical"
    assert classify(tok, single, digits) == "all-single-digit"
    assert classify(tok, greedy, digits) == "greedy-longest"
    # and the three are genuinely different segmentations
    assert len({tuple(canon), tuple(single), tuple(greedy)}) == 3


def test_summarise_fractions() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    digits = expansion(7, 12)
    single = [tok.encode(c, add_special_tokens=False)[0] for c in digits]
    out = summarise(tok, [(single, digits)])
    assert out["frac_single_digit_tokens"] == pytest.approx(1.0)
    assert out["attractor/all-single-digit"] == pytest.approx(1.0)
    assert out["mean_reward"] == pytest.approx(12.0)
