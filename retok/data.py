"""Reversed-addition data: arithmetic, encoding, streaming, and eval batches.

Task. The model reads ``<bos> [a] + [b] =`` where ``[a]``/``[b]`` are merged
D-digit tokens (canonically tokenized prompt — so their embeddings are
well-trained, not orphaned), then continues in one of two formats:

- **CoT**: emit the answer digit-by-digit, LSB-first: ``d0 d1 ... d_{D-1} <eos>``.
  Each digit gets its own position/forward-pass, so the carry can propagate
  serially across positions.
- **direct**: emit the whole answer as a single merged token ``[s] <eos>``.
  One position must resolve the entire carry chain — a parallel-depth task.

The two formats are mixed *without a cue* in the prompt, so the model's
distribution at ``=`` genuinely splits between them. A re-tokenized CoT answer
(``canonicalize`` collapses the D digit tokens into the merged token) is then
byte-identical to a direct-mode generation — that is the whole point.

Numbers are sampled so the sum stays within D digits (no overflow), which keeps
every answer collapsible to exactly one merged token and bounds the carry chain
to length ``D-1``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from common.streaming import SyntheticStream
from retok.tokenizer import RetokTokenizer

# Format codes (stored per-example for per-format metrics).
FMT_DIRECT = 0
FMT_COT = 1


def seq_len(n_digits: int) -> int:
    """Padded sequence length (CoT is the longest format).

    ``<bos>`` + D operand-a digits + ``+`` + D operand-b digits + ``=`` +
    D answer digits + ``<eos>`` = ``3*D + 4``.
    """
    return 3 * n_digits + 4


@dataclass(frozen=True)
class Arithmetic:
    """The carry structure of one reversed addition ``a + b = s``."""

    a: int
    b: int
    s: int
    carry_in: tuple[int, ...]  # carry entering each digit position (LSB-first)
    n_carries: int  # count of positions with an incoming carry
    chain_len: int  # longest run of consecutive incoming carries (serial depth)


def compute_arithmetic(a: int, b: int, n_digits: int, base: int = 10) -> Arithmetic:
    """Reversed long-addition carry structure of ``a + b`` in ``base``.

    Raises ValueError on overflow (sum needs more than ``n_digits`` digits),
    which the samplers avoid by construction.
    """
    a_digits = [(a // base**i) % base for i in range(n_digits)]
    b_digits = [(b // base**i) % base for i in range(n_digits)]
    carry_in: list[int] = []
    carry = 0
    for i in range(n_digits):
        carry_in.append(carry)
        total = a_digits[i] + b_digits[i] + carry
        carry = total // base
    if carry != 0:
        msg = f"overflow: {a} + {b} needs more than {n_digits} base-{base} digits"
        raise ValueError(msg)
    # Longest run of consecutive incoming carries = serial carry-chain depth.
    chain_len = 0
    run = 0
    for c in carry_in:
        run = run + 1 if c else 0
        chain_len = max(chain_len, run)
    return Arithmetic(
        a=a,
        b=b,
        s=a + b,
        carry_in=tuple(carry_in),
        n_carries=sum(carry_in),
        chain_len=chain_len,
    )


def sample_addition(
    rng: random.Random, n_digits: int, base: int = 10
) -> tuple[int, int]:
    """Sample ``(a, b)`` with uniform sum and uniform split, no overflow.

    Sampling ``s ~ U[0, base^D)`` then ``a ~ U[0, s]``, ``b = s - a`` gives a
    uniform answer and an unbiased operand split, so every answer digit is
    roughly uniform.
    """
    hi = base**n_digits - 1
    s = rng.randint(0, hi)
    a = rng.randint(0, s)
    return a, s - a


def sample_addition_with_chain(
    rng: random.Random,
    n_digits: int,
    chain_len: int,
    base: int = 10,
    max_attempts: int = 200_000,
) -> tuple[int, int]:
    """Rejection-sample ``(a, b)`` whose carry chain has exactly ``chain_len``.

    Used to build balanced per-difficulty eval buckets. Raises if a bucket
    can't be filled within ``max_attempts`` (a signal the bucket is empty for
    this ``n_digits``, e.g. chain_len >= n_digits).
    """
    for _ in range(max_attempts):
        a, b = sample_addition(rng, n_digits, base)
        if compute_arithmetic(a, b, n_digits, base).chain_len == chain_len:
            return a, b
    msg = (
        f"could not sample chain_len={chain_len} for n_digits={n_digits} "
        f"in {max_attempts} attempts"
    )
    raise RuntimeError(msg)


@dataclass
class Encoded:
    """One encoded example, padded to ``seq_len(n_digits)``."""

    input_ids: list[int]
    answer_mask: list[bool]  # positions whose prediction is trained/scored
    answer_token_positions: list[int]  # positions of answer tokens (excl. eos)
    fmt: int
    arithmetic: Arithmetic


def encode_example(tok: RetokTokenizer, a: int, b: int, fmt: int) -> Encoded:
    """Encode ``a + b`` in CoT or direct format, padded to ``seq_len``.

    Operands are given as **individual digit tokens** (LSB-first, like the
    answer) so the model can actually read the digits it must add — a merged
    operand token would hide them and make digit-wise computation impossible.
    The merged vocabulary is used only for the *answer* (direct format), which
    is the span a re-tokenized transcript collapses.

    **Caveat this creates:** a 3-digit operand run is itself mergeable, so the
    prompt is not canonical under :meth:`RetokTokenizer.canonicalize` — the full
    stream ``7 5 0 + 8 6 0 = 5 2 1`` canonicalizes to
    ``[750] + [860] = [521]``, merging the operands too. The analysis therefore
    treats the *direct format* (digit operands, merged answer) as the canonical
    replay rather than canonicalizing the whole sequence: the operands are
    conditioning context the model was given, not tokens it emitted, and holding
    them fixed isolates the answer-span collapse. Making the prompt genuinely
    canonical would need operand runs shorter than ``n_digits`` (e.g. 2-digit
    operands, 3-digit answers), which is a retrain. Note that canonicalizing a
    CoT example's generated answer digits yields exactly this function's direct
    encoding of the same ``(a, b)`` — "canonical replay" == "direct format".

    ``answer_mask`` marks the tokens the model must predict (answer tokens plus
    the terminating ``<eos>``); loss/accuracy use it via a causal shift, so the
    position that predicts token ``t`` is ``t-1``.
    """
    n_digits = tok.n_digits
    arith = compute_arithmetic(a, b, n_digits, tok.base)
    prompt = [
        tok.bos_id,
        *(tok.digit_token(d) for d in tok.rev_digits(a)),
        tok.plus_id,
        *(tok.digit_token(d) for d in tok.rev_digits(b)),
        tok.eq_id,
    ]
    answer_positions: list[int] = []
    if fmt == FMT_COT:
        answer = [tok.digit_token(d) for d in tok.rev_digits(arith.s)]
    else:
        answer = [tok.merged_token(arith.s)]
    ids = [*prompt, *answer, tok.eos_id]
    # answer tokens occupy indices len(prompt) .. len(prompt)+len(answer)-1
    for j in range(len(answer)):
        answer_positions.append(len(prompt) + j)
    eos_pos = len(prompt) + len(answer)
    mask = [False] * len(ids)
    for p in (*answer_positions, eos_pos):
        mask[p] = True
    # pad
    target_len = seq_len(n_digits)
    pad_n = target_len - len(ids)
    ids.extend([tok.pad_id] * pad_n)
    mask.extend([False] * pad_n)
    return Encoded(
        input_ids=ids,
        answer_mask=mask,
        answer_token_positions=answer_positions,
        fmt=fmt,
        arithmetic=arith,
    )


def _to_batch_item(enc: Encoded) -> dict[str, Tensor]:
    return {
        "input_ids": torch.tensor(enc.input_ids, dtype=torch.long),
        "answer_mask": torch.tensor(enc.answer_mask, dtype=torch.bool),
        "fmt": torch.tensor(enc.fmt, dtype=torch.long),
        "chain_len": torch.tensor(enc.arithmetic.chain_len, dtype=torch.long),
        "n_carries": torch.tensor(enc.arithmetic.n_carries, dtype=torch.long),
    }


def eq_position(n_digits: int) -> int:
    """Index of the ``=`` token: bos + D a-digits + '+' + D b-digits."""
    return 2 * n_digits + 2


def encode_one_step(tok: RetokTokenizer, a: int, b: int) -> dict[str, Tensor]:
    """Prompt-only encoding for the matched no-CoT (one-step) control.

    The input is just ``<bos> a… + b… =`` (padded); the target is the vector of
    D answer digit *values* (0..base-1), predicted in parallel from the ``=``
    position. No answer tokens are fed, so nothing leaks.
    """
    arith = compute_arithmetic(a, b, tok.n_digits, tok.base)
    prompt = [
        tok.bos_id,
        *(tok.digit_token(d) for d in tok.rev_digits(a)),
        tok.plus_id,
        *(tok.digit_token(d) for d in tok.rev_digits(b)),
        tok.eq_id,
    ]
    ids = prompt + [tok.pad_id] * (seq_len(tok.n_digits) - len(prompt))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "answer_digits": torch.tensor(tok.rev_digits(arith.s), dtype=torch.long),
        "chain_len": torch.tensor(arith.chain_len, dtype=torch.long),
        "n_carries": torch.tensor(arith.n_carries, dtype=torch.long),
    }


class ReversedAdditionDataset(SyntheticStream[dict[str, Tensor]]):
    """Fresh mixed-format reversed-addition examples generated on the fly.

    Built on :class:`common.streaming.SyntheticStream` for worker
    seeding/sharding and per-epoch seed mixing (the validity-critical plumbing).
    """

    def __init__(
        self,
        n_examples: int,
        *,
        n_digits: int = 4,
        base: int = 10,
        direct_fraction: float = 0.15,
        mode: str = "cot",
        seed: int = 42,
        tokenizer: RetokTokenizer | None = None,
    ) -> None:
        super().__init__(n_examples=n_examples, seed=seed)
        self.tok = tokenizer or RetokTokenizer(n_digits, base)
        self.n_digits = self.tok.n_digits
        self.base = self.tok.base
        self.direct_fraction = direct_fraction
        self.mode = mode

    def generate(self, rng: random.Random) -> dict[str, Tensor]:
        a, b = sample_addition(rng, self.n_digits, self.base)
        if self.mode == "one_step":
            return encode_one_step(self.tok, a, b)
        fmt = FMT_DIRECT if rng.random() < self.direct_fraction else FMT_COT
        return _to_batch_item(encode_example(self.tok, a, b, fmt))


def collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}


def make_eval_buckets(
    tok: RetokTokenizer,
    *,
    n_per_bucket: int,
    seed: int,
) -> dict[int, list[tuple[int, int]]]:
    """Sample unique ``(a, b)`` pairs bucketed by carry-chain length (0..D-1).

    The same pairs are later encoded in *both* formats so per-format accuracy is
    compared on identical arithmetic. These pairs are **in-distribution, not a
    held-out split**: the (a,b) space is small (~500k for D=3/base-10) and
    training streams tens of millions of examples, so eval pairs are seen during
    training. That is intentional — we characterize the trained model's
    CoT-vs-direct behaviour on its own distribution, not generalization to
    unseen pairs (and direct failing on *seen* pairs strengthens "can't do it in
    one step"). A separate RNG just makes the eval set reproducible.

    Some (base, n_digits, chain_len) buckets contain fewer than ``n_per_bucket``
    distinct pairs (e.g. the longest chain in a small base). Rather than loop
    forever we stop a bucket after too many consecutive duplicate draws and
    return what we found, logging the shortfall so it can't masquerade as full
    coverage.
    """
    rng = random.Random(f"retok-eval:{seed}")
    buckets: dict[int, list[tuple[int, int]]] = {}
    for chain_len in range(tok.n_digits):  # 0 .. D-1 (no overflow => max D-1)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        stall = 0
        while len(pairs) < n_per_bucket and stall < 20_000:
            try:
                a, b = sample_addition_with_chain(
                    rng, tok.n_digits, chain_len, tok.base, max_attempts=5_000
                )
            except RuntimeError:
                # This chain length is too rare to hit (e.g. the longest chain in
                # a large base) — leave the bucket short rather than crash.
                break
            if (a, b) in seen:
                stall += 1
                continue
            stall = 0
            seen.add((a, b))
            pairs.append((a, b))
        if len(pairs) < n_per_bucket:
            print(
                f">>> eval bucket chain_len={chain_len}: only {len(pairs)} distinct "
                f"pairs available (< {n_per_bucket} requested) for base={tok.base}, "
                f"n_digits={tok.n_digits}"
            )
        buckets[chain_len] = pairs
    return buckets


def canonical_replay_len(n_digits: int) -> int:
    """Length of the fully re-tokenized transcript: bos, [a], +, [b], =, [s], eos."""
    return 7


def encode_canonical_replay(
    tok: RetokTokenizer, pairs: list[tuple[int, int]]
) -> Tensor:
    """The stored transcript as a real encoder would re-read it.

    A CoT stream is *text*; storing and re-encoding it runs the tokenizer over
    the **whole** string, not just the span the model generated. Greedy
    longest-match therefore merges the operand runs as well as the answer::

        <bos> 7 5 0 + 8 6 0 = 5 2 1 <eos>   ->   <bos> [750] + [860] = [521] <eos>

    This is what :func:`RetokTokenizer.canonicalize` already does; earlier
    analysis exempted the prompt from it, which was not something a real
    tokenizer would do. 12 positions collapse to 7.

    Note the replay is **off-distribution**: merged tokens appear only after
    ``=`` in training, so the model has never read one as an operand. That is
    not a flaw in the measurement — it is the point. A re-tokenized transcript
    is a sequence the model never emitted and cannot consume. Where the question
    is specifically "could the model have done this in one step?", use
    ``FMT_DIRECT`` instead: it keeps the operands readable and so measures
    capacity rather than distribution shift.
    """
    out = []
    for a, b in pairs:
        ids = [
            t for t in encode_example(tok, a, b, FMT_COT).input_ids if t != tok.pad_id
        ]
        out.append(torch.tensor(tok.canonicalize(ids), dtype=torch.long))
    return torch.stack(out)


def encode_eval_batch(
    tok: RetokTokenizer, pairs: list[tuple[int, int]], fmt: int
) -> dict[str, Tensor]:
    """Encode a list of ``(a, b)`` pairs in one format into a stacked batch.

    Also returns per-example answer targets so scoring needs no re-derivation:
    ``answer_token_positions`` (B, n_answer) and ``answer_targets`` (B, n_answer).
    """
    encoded = [encode_example(tok, a, b, fmt) for a, b in pairs]
    n_answer = len(encoded[0].answer_token_positions)
    positions = torch.tensor(
        [e.answer_token_positions for e in encoded], dtype=torch.long
    )
    input_ids = torch.stack(
        [torch.tensor(e.input_ids, dtype=torch.long) for e in encoded]
    )
    # target token at each answer position
    targets = torch.stack([input_ids[i, positions[i]] for i in range(len(encoded))])
    return {
        "input_ids": input_ids,
        "answer_token_positions": positions,  # (B, n_answer)
        "answer_targets": targets,  # (B, n_answer)
        "n_answer": torch.tensor(n_answer, dtype=torch.long),
    }
