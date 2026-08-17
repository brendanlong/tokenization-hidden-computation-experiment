"""Tokenizer for the reversed-addition retokenization task.

The construction hinges on **BPE many-to-one decoding**: a multi-token digit
sequence and a single "merged" number token can decode to the *same* surface
string, so re-encoding the string recovers only the canonical (merged)
tokenization. We reproduce that with a hand-built vocabulary:

- ``base`` single-digit tokens ``0``..``base-1`` (used when the model emits an
  answer digit-by-digit, "CoT" style).
- one **merged token per D-digit string** (``base**D`` of them). A merged
  token's surface form is exactly the concatenation of its D digit characters,
  so the merged token for reversed ``"8010"`` decodes to the same string as the
  four digit tokens ``8 0 1 0``.

Everything lives in **reversed (least-significant-digit-first) convention**:
the model reads/writes numbers LSB-first so the carry propagates left-to-right,
and the merged answer token's surface is the *reversed* answer string. This is
the "string consistency under reversal" requirement — the merged token that a
re-tokenized CoT answer collapses to must be the token the direct-mode head
would emit, and both are keyed by the reversed surface string.

**Base is a free knob**, and an important one. The merged vocabulary has
``base**D`` tokens while the carry chain can be as long as ``D-1``. A small
base (e.g. binary) gives long carry chains — the serial-depth axis of the demo
— with a small, easily-trained merged vocabulary, decoupling "how hard is the
one-step retrieval" from "how long is the serial computation".

Canonicalization models a real BPE encoder's greedy longest-match: a maximal
run of D consecutive digit tokens is the longest vocabulary item spanning it,
so it collapses to the single merged token. See :func:`canonicalize`.
"""

from __future__ import annotations


class RetokTokenizer:
    """Vocabulary + (de)tokenization for D-digit base-``base`` reversed addition.

    Token layout (ids are stable given ``base`` and ``n_digits``)::

        0                      <pad>
        1 .. base              digits 0 .. base-1        (digit_offset = 1)
        base+1                 +
        base+2                 =
        base+3                 <bos>
        base+4                 <eos>
        base+5 .. +base**D     merged D-digit tokens (surface = reversed number)
    """

    def __init__(
        self, n_digits: int = 4, base: int = 10, *, include_merged: bool = True
    ) -> None:
        if n_digits < 1:
            msg = f"n_digits must be >= 1, got {n_digits}"
            raise ValueError(msg)
        if base < 2:
            msg = f"base must be >= 2, got {base}"
            raise ValueError(msg)
        self.n_digits = n_digits
        self.base = base
        self.include_merged = include_merged
        self.pad_id = 0
        self.digit_offset = 1  # digit d -> token 1 + d
        self.plus_id = base + 1
        self.eq_id = base + 2
        self.bos_id = base + 3
        self.eos_id = base + 4
        self.merged_base = base + 5
        # The one-step capability control never emits merged tokens, so it drops
        # them entirely — at large D, base**D merged tokens would be an
        # impossible embedding table (2**24 ≈ 16.7M). include_merged=False gives
        # a lean digit-only vocab.
        self.n_merged = base**n_digits if include_merged else 0
        self.vocab_size = self.merged_base + self.n_merged

    # --- digits ---------------------------------------------------------
    def digit_token(self, d: int) -> int:
        if not 0 <= d < self.base:
            msg = f"digit must be 0-{self.base - 1}, got {d}"
            raise ValueError(msg)
        return self.digit_offset + d

    def is_digit_token(self, token_id: int) -> bool:
        return self.digit_offset <= token_id < self.digit_offset + self.base

    def digit_value(self, token_id: int) -> int:
        if not self.is_digit_token(token_id):
            msg = f"not a digit token: {token_id}"
            raise ValueError(msg)
        return token_id - self.digit_offset

    # --- reversed-digit helpers ----------------------------------------
    def rev_digits(self, n: int) -> list[int]:
        """LSB-first digit list of ``n`` in ``base``, zero-padded to ``n_digits``.

        e.g. n=108, D=4, base=10 -> [8, 0, 1, 0] (surface "8010").
        """
        if not 0 <= n < self.base**self.n_digits:
            msg = f"n must be in [0, {self.base}^{self.n_digits}), got {n}"
            raise ValueError(msg)
        return [(n // self.base**i) % self.base for i in range(self.n_digits)]

    def _surface_int(self, rev_digit_list: list[int]) -> int:
        """Integer whose D-char base-``base`` string equals the reversed surface.

        The reversed digit list ``[8, 0, 1, 0]`` has surface string ``"8010"``,
        i.e. (in base 10) integer ``8010`` — the index used for the merged
        token. This is NOT the numeric value of the original number; it is the
        value of the *reversed* string, so operands and answers with the same
        reversed surface map to the same merged token.
        """
        return sum(
            d * self.base ** (self.n_digits - 1 - i)
            for i, d in enumerate(rev_digit_list)
        )

    # --- merged tokens --------------------------------------------------
    def merged_token(self, n: int) -> int:
        """Merged token for the reversed surface of number ``n``."""
        if not self.include_merged:
            raise ValueError("tokenizer built with include_merged=False")
        return self.merged_base + self._surface_int(self.rev_digits(n))

    def is_merged_token(self, token_id: int) -> bool:
        return self.merged_base <= token_id < self.merged_base + self.n_merged

    def merged_surface_digits(self, token_id: int) -> list[int]:
        """Digit list (in surface order) of a merged token."""
        if not self.is_merged_token(token_id):
            msg = f"not a merged token: {token_id}"
            raise ValueError(msg)
        value = token_id - self.merged_base
        # surface order = most-significant char of the D-char string first
        return [
            (value // self.base ** (self.n_digits - 1 - i)) % self.base
            for i in range(self.n_digits)
        ]

    def merged_token_for_digit_run(self, digit_tokens: list[int]) -> int:
        """Merged token whose surface equals this run of digit tokens.

        The digit tokens are in emission (surface) order. Requires exactly
        ``n_digits`` of them.
        """
        if len(digit_tokens) != self.n_digits:
            msg = f"expected {self.n_digits} digit tokens, got {len(digit_tokens)}"
            raise ValueError(msg)
        surface = [self.digit_value(t) for t in digit_tokens]
        return self.merged_base + sum(
            d * self.base ** (self.n_digits - 1 - i) for i, d in enumerate(surface)
        )

    # --- canonicalization ----------------------------------------------
    def canonicalize(self, token_ids: list[int]) -> list[int]:
        """Re-encode a token stream the way a greedy longest-match BPE would.

        Any maximal run of exactly ``n_digits`` consecutive digit tokens is the
        longest vocabulary item spanning it, so it collapses to the single
        merged token — modelling how a stored-then-re-tokenized transcript
        loses the digit-by-digit segmentation. Runs of a different length are
        left as digit tokens (no merged token of that length exists), which is
        why the task pads answers to exactly ``n_digits``. Non-digit tokens
        pass through unchanged.
        """
        out: list[int] = []
        i = 0
        n = len(token_ids)
        while i < n:
            if self.is_digit_token(token_ids[i]):
                j = i
                while j < n and self.is_digit_token(token_ids[j]):
                    j += 1
                run = token_ids[i:j]
                # Greedy longest match: chunk the run into D-digit merged tokens
                # from the left; a trailing remainder shorter than D stays as
                # digit tokens (no shorter merged token exists).
                k = 0
                while k + self.n_digits <= len(run):
                    out.append(
                        self.merged_token_for_digit_run(run[k : k + self.n_digits])
                    )
                    k += self.n_digits
                out.extend(run[k:])
                i = j
            else:
                out.append(token_ids[i])
                i += 1
        return out

    # --- pretty printing ------------------------------------------------
    def token_to_str(self, token_id: int) -> str:
        if token_id == self.pad_id:
            return "<pad>"
        if self.is_digit_token(token_id):
            return str(self.digit_value(token_id))
        if token_id == self.plus_id:
            return "+"
        if token_id == self.eq_id:
            return "="
        if token_id == self.bos_id:
            return "<bos>"
        if token_id == self.eos_id:
            return "<eos>"
        if self.is_merged_token(token_id):
            digits = "".join(str(d) for d in self.merged_surface_digits(token_id))
            return f"[{digits}]"
        msg = f"unknown token id: {token_id}"
        raise ValueError(msg)

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.token_to_str(t) for t in token_ids if t != self.pad_id)
