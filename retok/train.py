"""Train the reversed-addition mixture model for the retok experiment.

Usage:
    # Smoke test (tiny, CPU/GPU, no wandb)
    uv run python -m retok.train \
        --n-digits 4 --n-layers 2 --dim 128 \
        --generate-n 200000 --no-wandb --no-compile

    # Main run (streaming unique data, single epoch). The final checkpoint is
    # always written to --checkpoint-dir.
    uv run python -m retok.train \
        --n-digits 3 --n-layers 2 --dim 16 --generate-n 20000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.gpu import resolve_device
from retok.config import RetokTrainingConfig, retok_model_config
from retok.data import (
    ReversedAdditionDataset,
    collate,
    seq_len,
)
from retok.model import create_model
from retok.tokenizer import RetokTokenizer
from retok.training import train_retok_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train retok reversed-addition model")
    # Task
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--n-digits", type=int, default=3)
    parser.add_argument("--mode", default="cot", choices=["cot", "one_step"])
    parser.add_argument(
        "--no-merged",
        action="store_true",
        help=(
            "Drop the base**n_digits merged vocab. Required for large D "
            "(base**D embeddings are infeasible). Valid when no merged tokens "
            "are emitted — one_step (auto), or cot with --direct-fraction 0."
        ),
    )
    parser.add_argument("--direct-fraction", type=float, default=0.3)
    parser.add_argument(
        "--merged-operands",
        action="store_true",
        help="encode DIRECT-format operands as merged tokens (the fully "
        "re-tokenized transcript becomes in-distribution)",
    )
    parser.add_argument("--generate-n", type=int, default=20_000_000)
    # Model
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--init-std", type=float, default=None)
    # Optimization
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument(
        "--lr-schedule", default="cosine", choices=["cosine", "constant"]
    )
    # Eval / cadence
    parser.add_argument("--n-eval-per-bucket", type=int, default=2_000)
    parser.add_argument("--eval-batch-size", type=int, default=1_000)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--eval-every-steps", type=int, default=1_000)
    # Data loading
    parser.add_argument("--num-workers", type=int, default=4)
    # Infra
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="retok")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--checkpoint-dir", default="data/retok/checkpoints")
    args = parser.parse_args()

    config = RetokTrainingConfig(
        base=args.base,
        n_digits=args.n_digits,
        mode=args.mode,
        include_merged=not args.no_merged,
        direct_fraction=args.direct_fraction,
        merged_operands=args.merged_operands,
        generate_n=args.generate_n,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        init_std=args.init_std,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        n_eval_per_bucket=args.n_eval_per_bucket,
        log_every_steps=args.log_every_steps,
        eval_every_steps=args.eval_every_steps,
        seed=args.seed,
        no_compile=args.no_compile,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_dir=args.checkpoint_dir,
    )

    device = resolve_device()
    print(f"Device: {device}")
    torch.manual_seed(config.seed)

    tok = RetokTokenizer(
        config.n_digits, config.base, include_merged=config.include_merged
    )
    print(
        f"Tokenizer: base={config.base} n_digits={config.n_digits} "
        f"mode={config.mode} vocab_size={tok.vocab_size} "
        f"(merged tokens={tok.n_merged}, max carry-chain={config.n_digits - 1})"
    )

    run_name = config.wandb_run_name or (
        f"retok_{config.mode}_b{config.base}_d{config.n_digits}"
        f"_{config.n_layers}L_{config.dim}d"
    )

    model_config = retok_model_config(
        vocab_size=tok.vocab_size,
        max_seq_len=seq_len(config.n_digits),
        base=config.base,
        n_digits=config.n_digits,
        dim=config.dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        init_std=config.init_std,
    )
    model = create_model(model_config)

    dataset = ReversedAdditionDataset(
        config.generate_n,
        direct_fraction=config.direct_fraction,
        mode=config.mode,
        seed=config.seed + 100,
        tokenizer=tok,
        merged_operands=config.merged_operands,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    total_steps = config.generate_n // config.batch_size
    print(
        f"Streaming {config.generate_n:,} examples, single epoch "
        f"({total_steps:,} steps, batch_size={config.batch_size}, "
        f"direct_fraction={config.direct_fraction})"
    )

    result = train_retok_model(
        model=model,
        train_loader=train_loader,
        tok=tok,
        model_config=model_config,
        device=device,
        mode=config.mode,
        total_steps=total_steps,
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_schedule=config.lr_schedule,
        use_compile=not config.no_compile,
        n_eval_per_bucket=config.n_eval_per_bucket,
        eval_batch_size=args.eval_batch_size,
        log_every_steps=config.log_every_steps,
        eval_every_steps=config.eval_every_steps,
        seed=config.seed,
        run_name=run_name,
        # Per-run subdirectory, matching the published hf:checkpoints/<run>/
        # layout. (The flat layout let three seeds overwrite one final.pt,
        # 2026-08-30.)
        checkpoint_dir=Path(config.checkpoint_dir) / run_name,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_config={
            "model": model_config.model_dump(),
            "training": config.model_dump(),
        },
        merged_operands=config.merged_operands,
    )

    print("\n=== Final Evaluation ===")
    fe = result.final_eval
    if config.mode == "one_step":
        for chain_len in range(config.n_digits):
            acc = fe.get(f"eval/one_step_acc/chain_{chain_len}", 0.0)
            print(f"  chain_len={chain_len}: one_step(all D correct)={acc:.1%}")
        print("  per-digit-position accuracy (depth signature):")
        for i in range(config.n_digits):
            print(f"    pos {i}: {fe.get(f'eval/one_step_digit_acc/pos_{i}', 0.0):.1%}")
        print(f"  MEAN one_step={fe['eval/one_step_acc/mean']:.1%}")
    else:
        for chain_len in range(config.n_digits):
            cot = fe.get(f"eval/cot_acc/chain_{chain_len}", 0.0)
            direct = fe.get(f"eval/direct_acc/chain_{chain_len}", 0.0)
            p = fe.get(f"eval/direct_p_correct/chain_{chain_len}", 0.0)
            print(
                f"  chain_len={chain_len}: CoT={cot:.1%}  direct={direct:.1%}  "
                f"direct P(correct)={p:.3f}"
            )
        print(
            f"  MEAN: CoT={fe['eval/cot_acc/mean']:.1%}  "
            f"direct={fe['eval/direct_acc/mean']:.1%}"
        )


if __name__ == "__main__":
    main()
