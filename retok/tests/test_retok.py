"""Fast CPU tests for the retok experiment.

Cover the correctness invariants the whole demo rests on:
- the reversed-string / merged-token bijection is consistent between operands,
  answers, and canonicalization;
- **the core claim**: a CoT answer and its direct-mode counterpart decode to the
  same string, and canonicalizing the CoT token stream yields exactly the
  direct-mode (merged) token stream;
- carry arithmetic is correct;
- the streaming dataset is deterministic and worker-sharded without duplication;
- model forward-pass shapes.
"""

from __future__ import annotations

import random

import pytest
import torch
from torch.utils.data import DataLoader

from retok.config import retok_model_config
from retok.data import (
    FMT_COT,
    FMT_DIRECT,
    ReversedAdditionDataset,
    collate,
    compute_arithmetic,
    encode_example,
    encode_one_step,
    eq_position,
    make_eval_buckets,
    sample_addition,
    sample_addition_with_chain,
    seq_len,
)
from retok.model import create_model
from retok.tokenizer import RetokTokenizer


def test_merged_token_bijection() -> None:
    tok = RetokTokenizer(4)
    # distinct merged tokens for distinct numbers, and round-trip via surface
    seen = set()
    for n in range(0, 10_000, 137):
        t = tok.merged_token(n)
        assert tok.is_merged_token(t)
        assert t not in seen
        seen.add(t)
        # surface digits of the merged token == reversed digit list of n
        assert tok.merged_surface_digits(t) == tok.rev_digits(n)


def test_reversed_convention_and_canonical_collapse() -> None:
    """The whole construction: canonicalizing a CoT answer's digit span yields
    exactly the direct-format (merged) token — so "canonical replay" of a CoT
    generation is byte-identical to a genuine direct generation."""
    tok = RetokTokenizer(4)
    plen = 2 * 4 + 3  # bos + 4 a-digits + '+' + 4 b-digits + '='
    for a, b in [(99, 9), (0, 0), (4567, 5432), (1234, 8765), (500, 500)]:
        cot = encode_example(tok, a, b, FMT_COT)
        direct = encode_example(tok, a, b, FMT_DIRECT)
        # prompts (digit operands) identical
        assert cot.input_ids[:plen] == direct.input_ids[:plen]
        cot_unpadded = [t for t in cot.input_ids if t != tok.pad_id]
        direct_unpadded = [t for t in direct.input_ids if t != tok.pad_id]
        # answer digits canonicalize to the direct merged answer token
        answer_digits = cot_unpadded[plen : plen + 4]
        assert tok.canonicalize(answer_digits) == [direct_unpadded[plen]]
        # "canonical replay" = keep the (digit) prompt, re-tokenize only the
        # generated answer span == the direct-format sequence exactly
        replay = cot_unpadded[:plen] + tok.canonicalize(cot_unpadded[plen:])
        assert replay == direct_unpadded
        # full-stream canonicalize merges operands AND answer (3 merged tokens)
        full = tok.canonicalize(cot_unpadded)
        assert full == [
            tok.bos_id,
            tok.merged_token(a),
            tok.plus_id,
            tok.merged_token(b),
            tok.eq_id,
            tok.merged_token(a + b),
            tok.eos_id,
        ]


def test_canonicalize_only_collapses_full_runs() -> None:
    tok = RetokTokenizer(4)
    # a run shorter than D stays as digit tokens (no shorter merged token)
    three = [tok.digit_token(d) for d in (1, 2, 3)]
    assert tok.canonicalize([tok.bos_id, *three, tok.eos_id]) == [
        tok.bos_id,
        *three,
        tok.eos_id,
    ]
    # non-digit tokens pass through
    assert tok.canonicalize([tok.plus_id, tok.eq_id]) == [tok.plus_id, tok.eq_id]


def test_base_2_tokenizer_and_collapse() -> None:
    """Binary base: long carry chains with a tiny merged vocab. The collapse
    invariant must hold for any base."""
    tok = RetokTokenizer(n_digits=8, base=2)
    assert tok.n_merged == 2**8  # 256 merged tokens
    assert tok.base == 2
    plen = 2 * 8 + 3
    for a, b in [(0, 0), (255, 0), (170, 85), (127, 128), (1, 254)]:
        assert a + b <= 255
        cot = encode_example(tok, a, b, FMT_COT)
        direct = encode_example(tok, a, b, FMT_DIRECT)
        cot_u = [t for t in cot.input_ids if t != tok.pad_id]
        direct_u = [t for t in direct.input_ids if t != tok.pad_id]
        assert tok.canonicalize(cot_u[plen:]) == direct_u[plen:]
        # binary digits only 0/1
        assert all(tok.digit_value(t) in (0, 1) for t in cot_u[plen : plen + 8])


def test_base_2_long_carry_chain() -> None:
    # 0b01111111 + 0b00000001 = 0b10000000: carry propagates through 7 positions
    arith = compute_arithmetic(127, 1, 8, base=2)
    assert arith.s == 128
    assert arith.chain_len == 7  # long serial dependency, tiny (256) vocab


def test_arithmetic_carries() -> None:
    # 0999 + 0001 = 1000: carries propagate through positions 1,2,3 (chain 3)
    arith = compute_arithmetic(999, 1, 4)
    assert arith.s == 1000
    assert arith.carry_in == (0, 1, 1, 1)
    assert arith.n_carries == 3
    assert arith.chain_len == 3
    # no carries
    arith2 = compute_arithmetic(1111, 1111, 4)
    assert arith2.carry_in == (0, 0, 0, 0)
    assert arith2.chain_len == 0


def test_sample_addition_no_overflow() -> None:
    rng = random.Random(0)
    for _ in range(2000):
        a, b = sample_addition(rng, 4)
        assert 0 <= a + b <= 9999
        compute_arithmetic(a, b, 4)  # must not raise (no overflow)


def test_sample_addition_with_chain() -> None:
    rng = random.Random(1)
    for chain_len in range(4):
        a, b = sample_addition_with_chain(rng, 4, chain_len)
        assert compute_arithmetic(a, b, 4).chain_len == chain_len


def test_eval_buckets_balanced_and_disjoint_from_answers() -> None:
    tok = RetokTokenizer(4)
    buckets = make_eval_buckets(tok, n_per_bucket=50, seed=7)
    assert set(buckets) == set(range(4))  # chain lengths 0..3
    for chain_len, pairs in buckets.items():
        assert len(pairs) == 50
        assert len(set(pairs)) == 50  # no duplicate pairs within a bucket
        for a, b in pairs:
            assert compute_arithmetic(a, b, 4).chain_len == chain_len


def test_encoding_shapes_and_mask() -> None:
    tok = RetokTokenizer(4)
    enc = encode_example(tok, 1234, 4321, FMT_COT)
    assert len(enc.input_ids) == seq_len(4)
    assert len(enc.answer_mask) == seq_len(4)
    # CoT: 4 digit tokens + eos are masked as answer targets
    assert sum(enc.answer_mask) == 5
    direct = encode_example(tok, 1234, 4321, FMT_DIRECT)
    # direct: merged token + eos
    assert sum(direct.answer_mask) == 2


def test_dataset_deterministic() -> None:
    ds1 = ReversedAdditionDataset(64, n_digits=4, seed=123)
    ds2 = ReversedAdditionDataset(64, n_digits=4, seed=123)
    items1 = [item["input_ids"].tolist() for item in ds1]
    items2 = [item["input_ids"].tolist() for item in ds2]
    assert items1 == items2
    # different seed => different data
    ds3 = ReversedAdditionDataset(64, n_digits=4, seed=124)
    items3 = [item["input_ids"].tolist() for item in ds3]
    assert items1 != items3


def test_dataset_worker_sharding_no_duplication() -> None:
    """With multiple workers, one pass yields n unique examples, not n*workers
    duplicated copies (the SyntheticStream sharding invariant)."""
    n = 256
    ds = ReversedAdditionDataset(n, n_digits=4, seed=99)
    loader = DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=2)
    rows = [tuple(r) for batch in loader for r in batch["input_ids"].tolist()]
    assert len(rows) == n  # exactly n across workers (sharded, not duplicated)


def test_direct_fraction_mix() -> None:
    ds = ReversedAdditionDataset(4000, n_digits=4, direct_fraction=0.3, seed=5)
    fmts = [int(item["fmt"]) for item in ds]
    frac = sum(f == FMT_DIRECT for f in fmts) / len(fmts)
    assert 0.25 < frac < 0.35
    assert set(fmts) <= {FMT_DIRECT, FMT_COT}


def test_encode_one_step_no_leakage() -> None:
    tok = RetokTokenizer(n_digits=8, base=2)
    item = encode_one_step(tok, 127, 1)  # 127 + 1 = 128, chain 7
    ids = item["input_ids"].tolist()
    # everything from the "=" onward must be padding (answer not fed)
    eq = eq_position(8)
    assert ids[eq] == tok.eq_id
    assert all(t == tok.pad_id for t in ids[eq + 1 :])
    # target digits = reversed digits of 128
    assert item["answer_digits"].tolist() == tok.rev_digits(128)
    assert int(item["chain_len"]) == 7


def test_model_forward_shapes() -> None:
    tok = RetokTokenizer(3)  # smaller vocab for a fast test
    cfg = retok_model_config(
        vocab_size=tok.vocab_size,
        max_seq_len=seq_len(3),
        base=10,
        n_digits=3,
        dim=32,
        n_heads=2,
        n_layers=2,
    )
    model = create_model(cfg)
    ids = torch.randint(0, tok.vocab_size, (4, seq_len(3)))
    logits = model(ids)
    assert logits.shape == (4, seq_len(3), tok.vocab_size)
    logits2, residuals = model.forward_with_residuals(ids)
    assert torch.allclose(logits, logits2)
    assert len(residuals) == 2
    assert residuals[0].shape == (4, seq_len(3), 32)
    # parallel one-step readout: (B, n_digits, base) at the "=" position
    one_step = model.one_step_logits(ids, eq_position(3))
    assert one_step.shape == (4, 3, 10)


# --- Phase-2 per-token attribution ------------------------------------------


def _rate_stats(pairs: list[tuple[list[int], list[int]]]) -> dict[str, float]:
    """Aggregate (actual, canonical) token-id pairs the way phase2_probe does."""
    import difflib

    n = len(pairs)
    toks = sum(len(a) for a, _ in pairs)
    n_nc = 0
    bad = 0
    for actual, canon in pairs:
        if canon == actual:
            continue
        n_nc += 1
        bad += sum(
            i2 - i1
            for op, i1, i2, _, _ in difflib.SequenceMatcher(
                a=actual, b=canon, autojunk=False
            ).get_opcodes()
            if op != "equal"
        )
    return {
        "per_generation": n_nc / n,
        "mean_len": toks / n,
        "affected_per_nc": bad / max(1, n_nc),
        "per_token": bad / toks,
    }


def test_per_token_identity() -> None:
    """per_token == per_generation * affected_per_nc_gen / mean_len.

    The two rates are reported side by side in RESULTS.md precisely so they can
    be cross-checked; this pins the identity that relates them. It also catches
    the naive attribution (boolean / token count), which does NOT satisfy it.
    """
    pairs = [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),  # canonical
        ([1, 2, 3, 4, 5], [1, 9, 9, 4, 5]),  # 2 actual tokens changed
        ([6, 7, 8], [6, 7, 8]),  # canonical
        ([1, 2, 3, 4], [1, 22, 4]),  # 2 actual tokens -> 1 canonical
    ]
    st = _rate_stats(pairs)
    assert st["per_token"] == pytest.approx(
        st["per_generation"] * st["affected_per_nc"] / st["mean_len"]
    )
    # 4 affected actual tokens out of 17 emitted
    assert st["per_token"] == pytest.approx(4 / 17)
    # The naive version disagrees, which is the whole point of stating the identity.
    naive = st["per_generation"] / st["mean_len"]
    assert naive != pytest.approx(st["per_token"])


def test_autojunk_disabled_matters() -> None:
    """SequenceMatcher's autojunk silently corrupts diffs on long token streams.

    Any element in >1% of a sequence of length >=200 is treated as junk and
    cannot anchor a match. Token streams are full of such elements (spaces,
    common words), so the default would misattribute the rate.
    """
    import difflib

    common = 7  # stands in for a frequent token like " the"
    actual = [common] * 300 + [1, 2] + [common] * 10
    canon = [common] * 300 + [99] + [common] * 10

    def changed(autojunk: bool) -> int:
        return sum(
            i2 - i1
            for op, i1, i2, _, _ in difflib.SequenceMatcher(
                a=actual, b=canon, autojunk=autojunk
            ).get_opcodes()
            if op != "equal"
        )

    assert changed(autojunk=False) == 2  # exactly the two substituted tokens
    assert changed(autojunk=True) > 2  # junk heuristic over-attributes


def test_merged_operands_direct_equals_canonicalized_cot() -> None:
    """With merged operands, the direct encoding IS the canonical replay.

    This is the point of the flag: canonicalize() of the full CoT stream
    (operands included) must coincide token-for-token with the direct-format
    training encoding, making the fully re-tokenized transcript an
    in-distribution example rather than a sequence the model never read.
    """
    tok = RetokTokenizer(n_digits=3, base=10)
    for a, b in ((57, 68), (0, 0), (123, 456), (999, 0)):
        cot = encode_example(tok, a, b, FMT_COT)
        cot_ids = [t for t in cot.input_ids if t != tok.pad_id]
        direct = encode_example(tok, a, b, FMT_DIRECT, merged_operands=True)
        direct_ids = [t for t in direct.input_ids if t != tok.pad_id]
        assert tok.canonicalize(cot_ids) == direct_ids
        # CoT encoding itself is unchanged by the flag
        assert (
            encode_example(tok, a, b, FMT_COT, merged_operands=True).input_ids
            == cot.input_ids
        )
