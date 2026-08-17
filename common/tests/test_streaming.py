"""Tests for the validity-critical streaming plumbing.

These cover the exact bug classes that recurred across experiments:
worker duplication, stale per-epoch seeds, and train/val leakage when
splitting a single-source stream across DataLoader workers.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from common.streaming import (
    SyntheticStream,
    stream_rng,
    worker_share,
)

if TYPE_CHECKING:
    import random
    from pathlib import Path


class _IntStream(SyntheticStream[torch.Tensor]):
    def generate(self, rng: random.Random) -> torch.Tensor:
        return torch.tensor(rng.randrange(2**60))


def test_stream_rng_deterministic() -> None:
    a = stream_rng(42, epoch=3, worker_id=1).random()
    b = stream_rng(42, epoch=3, worker_id=1).random()
    assert a == b


def test_stream_rng_no_additive_collisions() -> None:
    # seed+epoch+worker_id additive seeding would make these collide
    variants = [
        stream_rng(42, epoch=1, worker_id=0),
        stream_rng(43, epoch=0, worker_id=0),
        stream_rng(42, epoch=0, worker_id=1),
    ]
    draws = [rng.random() for rng in variants]
    assert len(set(draws)) == len(draws)


def test_worker_share_sums_to_n() -> None:
    for n in (0, 1, 7, 100, 101):
        for num_workers in (1, 2, 3, 4):
            shares = [worker_share(n, num_workers, w) for w in range(num_workers)]
            assert sum(shares) == n
            assert max(shares) - min(shares) <= 1


def test_buffered_shuffle_is_a_permutation() -> None:
    from common.streaming import buffered_shuffle

    items = list(range(100))
    out = list(buffered_shuffle(iter(items), stream_rng(0), buffer_size=10))
    assert sorted(out) == items
    assert out != items


def test_synthetic_stream_yields_exact_count() -> None:
    ds = _IntStream(n_examples=37, seed=0)
    assert len(list(ds)) == 37


def test_synthetic_stream_fresh_data_per_epoch() -> None:
    ds = _IntStream(n_examples=20, seed=0)
    first = [t.item() for t in ds]
    second = [t.item() for t in ds]
    assert first != second
    ds.set_epoch(0)
    again = [t.item() for t in ds]
    assert first == again


def test_synthetic_stream_workers_no_duplicates() -> None:
    # The classic IterableDataset trap: without worker-aware seeding and
    # sharding, two workers each yield the full identical stream.
    ds = _IntStream(n_examples=101, seed=0)
    loader = DataLoader(ds, batch_size=None, num_workers=2)
    values = [int(t.item()) for t in loader]
    assert len(values) == 101
    assert len(set(values)) == 101


class _FakeTokenizer:
    """Maps "doc<K>" -> [K] so packed windows reveal document identity.

    Accepts either a single string or a batch (list) of strings, mirroring an
    HF tokenizer: a batched call returns a list of per-document id lists.
    """

    eos_token_id = 1_000_000

    def __call__(self, text: str | list[str]) -> SimpleNamespace:
        if isinstance(text, list):
            return SimpleNamespace(
                input_ids=[[int(t.removeprefix("doc"))] for t in text]
            )
        return SimpleNamespace(input_ids=[int(text.removeprefix("doc"))])


def _write_corpus(path: Path, n_docs: int) -> str:
    file = path / "corpus.jsonl"
    with file.open("w") as f:
        for i in range(n_docs):
            f.write(json.dumps({"text": f"doc{i}"}) + "\n")
    return str(file)
