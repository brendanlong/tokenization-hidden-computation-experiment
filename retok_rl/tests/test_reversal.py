"""CPU tests for the reversal task definition and its metrics."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from retok_rl.reversal import (
    WORDS_BY_LEN,
    build_reversal,
    classify_letters,
    letter_overlap,
    letter_prefix,
    split_words,
    summarise_reversal,
    tokens_covering_letters,
)


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    # gpt2 is ungated and byte-level BPE like Qwen's; the metric code is
    # tokenizer-generic.
    return AutoTokenizer.from_pretrained("gpt2")


def test_word_list_is_well_formed() -> None:
    seen: set[str] = set()
    for length, words in WORDS_BY_LEN.items():
        for w in words:
            assert len(w) == length, (length, w)
            assert w.isalpha() and w.islower(), w
            assert w not in seen, f"duplicate: {w}"
            seen.add(w)


def test_word_split_is_disjoint_and_stratified() -> None:
    train, held = split_words(60, seed=0)
    all_words = {w for ws in WORDS_BY_LEN.values() for w in ws}
    assert not set(train) & set(held)
    assert set(train) | set(held) == all_words
    # every length represented in the held-out set
    assert {len(w) for w in held} == set(WORDS_BY_LEN)


def test_prompt_contains_word_and_target_is_reversed(
    tok: PreTrainedTokenizerBase,
) -> None:
    for ex in build_reversal(("cat", "keyboard"), tok):
        assert ex.word in ex.prompt
        assert ex.target == ex.word[::-1]
        assert not ex.prompt.endswith(" ")


def test_letter_prefix() -> None:
    assert letter_prefix("  tac!") == "tac"
    assert letter_prefix("TAC, yes") == "tac"
    assert letter_prefix("123abc") == ""
    assert letter_prefix("") == ""


def test_letter_overlap() -> None:
    assert letter_overlap("tac", "tac") == 1.0
    assert letter_overlap("cat", "tac") == 1.0  # right letters, wrong order
    assert letter_overlap("xyz", "tac") == 0.0
    assert letter_overlap("ta", "tac") == pytest.approx(2 / 3)


def test_tokens_covering_letters_stops_at_non_letter(
    tok: PreTrainedTokenizerBase,
) -> None:
    ids = tok.encode(" tac, and more", add_special_tokens=False)
    kept, surface = tokens_covering_letters(tok, ids)
    assert surface == "tac"
    assert tok.decode(kept).strip() == "tac"


def test_classify_letters_attractors(tok: PreTrainedTokenizerBase) -> None:
    word = "esuohthgil"  # 'lighthouse' reversed
    single = [tok.encode(c, add_special_tokens=False) for c in word]
    single_ids = [t for enc in single for t in enc]
    assert classify_letters(tok, single_ids, word) == "all-single-char"

    canon_ids = tok.encode(word, add_special_tokens=False)
    label = classify_letters(tok, canon_ids, word)
    # canonical may coincide with another attractor only if segmentations
    # collide; for this word they don't:
    assert label == "canonical"

    # a mixture that matches none of the three
    mixed = tok.encode("esu", add_special_tokens=False) + [
        t for c in "ohthgil" for t in tok.encode(c, add_special_tokens=False)
    ]
    assert classify_letters(tok, mixed, word) in {"other", "greedy-longest"}
    assert classify_letters(tok, canon_ids, "different") == "other"


def test_summarise_reversal_correctness_metrics(tok: PreTrainedTokenizerBase) -> None:
    target = "tac"
    right = tok.encode("tac", add_special_tokens=False)
    wrong = tok.encode("dog", add_special_tokens=False)
    stats = summarise_reversal(tok, [(right, target), (wrong, target), ([], target)])
    assert stats["mean_reward"] == pytest.approx(3 / 3)  # 3 + 0 + 0 over 3
    assert stats["exact_match"] == pytest.approx(1 / 3)
    assert stats["attempted"] == pytest.approx(2 / 3)
    assert stats["empty"] == pytest.approx(1 / 3)
    assert 0.0 <= stats["letter_overlap"] <= 1.0
    assert stats["frac_single_char_tokens"] >= 0.0
    # both emissions are canonical encodings of their own text: wrong text is
    # just wrong — it must not register as non-canonical tokenization
    assert stats["roundtrip/gen_non_canonical"] == 0.0
    assert stats["roundtrip/tok_non_canonical"] == 0.0
    # segmentation-of-answer is computed over the compliant rollout only
    assert stats["n_compliant"] == 1.0
    assert stats["attractor_compliant/canonical"] == pytest.approx(1.0)


def test_summarise_reversal_roundtrip_flags_non_canonical(
    tok: PreTrainedTokenizerBase,
) -> None:
    target = "tac"
    # emit the correct answer one letter per token — non-canonical for gpt2
    single_ids = [t for c in "tac" for t in tok.encode(c, add_special_tokens=False)]
    assert len(single_ids) == 3
    stats = summarise_reversal(tok, [(single_ids, target)])
    assert stats["exact_match"] == 1.0
    assert stats["roundtrip/gen_non_canonical"] == 1.0
    assert stats["roundtrip/answer_non_canonical"] == 1.0
    assert stats["attractor_compliant/all-single-char"] == pytest.approx(1.0)
