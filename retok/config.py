"""Configuration for the retokenization (hidden-composition) experiment.

Two pydantic configs:

- :class:`RetokModelConfig` — architecture (no defaults tied to a specific
  run; build via :func:`retok_model_config`).
- :class:`RetokTrainingConfig` — training hyperparameters, extending the shared
  :class:`common.config.BaseTrainingConfig`. Defaults are the best-known
  settings so callers get good results without overrides.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

from common.config import BaseTrainingConfig

TrainMode = Literal["cot", "one_step"]


class RetokModelConfig(BaseModel):
    """Decoder-only transformer architecture for reversed addition.

    ``base`` and ``n_digits`` are carried here (not just in the training config)
    so a checkpoint is self-describing — analysis can rebuild the tokenizer and
    size the parallel one-step readout head without being told the task shape.
    """

    dim: int
    n_heads: int
    n_layers: int
    intermediate_dim: int
    vocab_size: int
    max_seq_len: int
    base: int
    n_digits: int
    dropout: float = 0.0
    norm_eps: float = 1e-5
    # None => PyTorch default init. 0.02 is standard for LLMs; small models on
    # this task train fine with the default, so leave it None unless sweeping.
    init_std: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> RetokModelConfig:
        if self.dim % self.n_heads != 0:
            msg = f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


def retok_model_config(
    *,
    vocab_size: int,
    max_seq_len: int,
    base: int,
    n_digits: int,
    dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.0,
    init_std: float | None = None,
) -> RetokModelConfig:
    """Build a :class:`RetokModelConfig` with best-known defaults.

    The default depth (``n_layers=2``) is deliberately shallow: the whole demo
    hinges on digit-by-digit CoT succeeding while a single-token ("direct")
    answer cannot resolve the full carry chain in one position. A shallow model
    opens that gap; the depth sweep (see EXPERIMENT_PLAN.md) picks the final
    value.
    """
    return RetokModelConfig(
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        intermediate_dim=dim * 4,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        base=base,
        n_digits=n_digits,
        dropout=dropout,
        init_std=init_std,
    )


class RetokTrainingConfig(BaseTrainingConfig):
    """Training hyperparameters for the retokenization experiment."""

    # --- Task ---
    # base + n_digits set the merged vocabulary (base**n_digits) and the max
    # carry-chain length (n_digits-1). A small base gives long carry chains (the
    # serial-depth axis) with a small, easily-trained merged vocabulary — see
    # the tokenizer docstring. Defaults picked by the depth/base sweep.
    base: int = 10
    n_digits: int = 3
    # Training/eval mode:
    #   "cot"      — autoregressive digit-by-digit (+ a direct_fraction of
    #                single merged-token answers); the serial computation.
    #   "one_step" — matched no-CoT control: predict ALL D answer digits in
    #                parallel from the "=" position via per-digit heads. Tests
    #                one-step capability with NO merged-vocab retrieval wall, so
    #                a shallow model's failure on long carry chains reflects a
    #                depth limit, not a retrieval artifact.
    mode: TrainMode = "cot"
    # Whether the vocabulary includes the base**n_digits merged tokens. Needed
    # for CoT/direct (the collapse); droppable for the one_step control, which
    # never emits them — essential at large D where base**D is an impossible
    # embedding table. Defaults to False in one_step mode (see the validator).
    include_merged: bool = True
    # Fraction of training examples emitted in "direct" (single merged token)
    # format rather than digit-by-digit CoT. The rest are CoT. An uncued
    # mixture: the prompt does NOT signal which format follows, so the model's
    # distribution at "=" genuinely splits between the two — which is what makes
    # a re-tokenized CoT transcript a bona fide sample from the model.
    direct_fraction: float = 0.3
    # Direct-format examples encode operands as merged tokens too, making the
    # fully re-tokenized CoT transcript an in-distribution format. CoT
    # examples are unchanged. See encode_example(merged_operands=...).
    merged_operands: bool = False

    # --- Streaming data (always single-epoch unique data) ---
    generate_n: int = 20_000_000
    n_epochs: int = 1

    # --- Eval ---
    # Held-out examples per carry-count bucket, evaluated in BOTH formats.
    n_eval_per_bucket: int = 2_000

    # --- Model architecture (mirrored into RetokModelConfig) ---
    # dim=16 is the locked regime: CoT succeeds but the single-token direct form
    # genuinely fails (see EXPERIMENT_PLAN + width-sweep appendix). Wider models
    # can one-pass it, defeating the claim.
    dim: int = 16
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    init_std: float | None = None

    # --- Optimization (override BaseTrainingConfig defaults) ---
    batch_size: int = 512
    lr: float = 3e-4
    warmup_steps: int = 200

    # --- Cadence ---
    log_every_steps: int = 100
    eval_every_steps: int = 1_000

    # --- Misc ---
    no_compile: bool = False

    # --- Wandb ---
    wandb_project: str = "retok"

    @model_validator(mode="after")
    def _validate(self) -> RetokTrainingConfig:
        if not 0.0 <= self.direct_fraction <= 1.0:
            msg = f"direct_fraction must be in [0, 1], got {self.direct_fraction}"
            raise ValueError(msg)
        if self.n_digits < 1:
            msg = f"n_digits must be >= 1, got {self.n_digits}"
            raise ValueError(msg)
        # The one-step control never emits merged tokens; drop them so large D
        # (long carry chains) stays feasible instead of allocating base**D
        # embeddings. CoT/direct keep them (the collapse needs merged tokens).
        if self.mode == "one_step":
            self.include_merged = False
        if not self.include_merged and self.mode == "cot" and self.direct_fraction > 0:
            msg = (
                "include_merged=False requires no merged-token emission: use "
                "mode=one_step, or cot with direct_fraction=0."
            )
            raise ValueError(msg)
        return self
