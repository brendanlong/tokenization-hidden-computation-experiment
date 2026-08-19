"""Re-evaluate published width-sweep checkpoints — CPU only, no training.

The appendix width-sweep table (WRITEUP.md / RESULTS.md) has three columns per
width: CoT, one-step parallel per-digit, and one-step merged-token ("direct").
The CoT and direct columns come from the ``retok-sweep-cot-d*`` mixture
checkpoints and the per-digit column from the ``retok-sweep-1step-d*``
checkpoints, which drop the merged vocabulary entirely (``include_merged=False``)
— which is why ``retok.analysis`` rejects them. This evaluates either kind,
detected from the checkpoint's own vocab size.

There is no ``retok-sweep-cot-d16``: that arm is the headline model itself
(``retok-main-s0``, trained on 20M examples where the sweep used 15M). Expect
that substitution to show: the dim-16 *direct (merged token)* cell reads
~0.8-1.0% here against the sweep table's published 1.5% — a different
checkpoint, evaluated at CPU/fp32 where the original eval ran bf16 on GPU, and
~1%-accuracy argmax cells are precision-sensitive at exactly that scale. The
other eleven cells reproduce within a few tenths of a point.

Run (downloads each ~0.1 MB checkpoint from the published dataset on demand):

    uv run python -m retok.eval_sweep            # the full published sweep
    uv run python -m retok.eval_sweep \\
        --checkpoints hf:checkpoints/retok-sweep-1step-d64/final.pt
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext

# dim -> (cot/direct checkpoint, one-step checkpoint); d16's mixture arm is the
# headline run.
SWEEP_CHECKPOINTS = [
    "hf:checkpoints/retok-sweep-cot-d8/final.pt",
    "hf:checkpoints/retok-sweep-cot-d12/final.pt",
    "hf:checkpoints/retok-main-s0/final.pt",
    "hf:checkpoints/retok-sweep-cot-d24/final.pt",
    "hf:checkpoints/retok-sweep-cot-d32/final.pt",
    "hf:checkpoints/retok-sweep-cot-d64/final.pt",
    "hf:checkpoints/retok-sweep-1step-d8/final.pt",
    "hf:checkpoints/retok-sweep-1step-d12/final.pt",
    "hf:checkpoints/retok-sweep-1step-d16/final.pt",
    "hf:checkpoints/retok-sweep-1step-d24/final.pt",
    "hf:checkpoints/retok-sweep-1step-d32/final.pt",
    "hf:checkpoints/retok-sweep-1step-d64/final.pt",
]


def evaluate_checkpoint(
    checkpoint: str, *, n_per_bucket: int, seed: int
) -> dict[str, float]:
    from common.checkpoint import resolve_checkpoint
    from common.gpu import resolve_device
    from retok.data import make_eval_buckets
    from retok.tokenizer import RetokTokenizer
    from retok.training import evaluate, evaluate_one_step, load_model

    device = resolve_device(require_cuda=False)
    model, cfg = load_model(resolve_checkpoint(checkpoint), device)
    mixture_tok = RetokTokenizer(cfg.n_digits, cfg.base)
    one_step_tok = RetokTokenizer(cfg.n_digits, cfg.base, include_merged=False)
    if cfg.vocab_size == mixture_tok.vocab_size:
        tok, kind = mixture_tok, "mixture"
    elif cfg.vocab_size == one_step_tok.vocab_size:
        tok, kind = one_step_tok, "one_step"
    else:
        raise ValueError(
            f"{checkpoint}: vocab {cfg.vocab_size} matches neither the mixture "
            f"({mixture_tok.vocab_size}) nor merged-free ({one_step_tok.vocab_size}) "
            "tokenizer"
        )
    buckets = make_eval_buckets(tok, n_per_bucket=n_per_bucket, seed=seed)
    eval_fn = evaluate if kind == "mixture" else evaluate_one_step
    metrics = eval_fn(
        model, tok, buckets, device, eval_batch_size=1000, autocast_ctx=nullcontext()
    )
    dim = getattr(cfg, "dim", getattr(cfg, "d_model", "?"))
    if kind == "mixture":
        print(
            f"{checkpoint}\n  dim={dim}  "
            f"CoT={metrics['eval/cot_acc/mean']:.1%}  "
            f"direct (merged token)={metrics['eval/direct_acc/mean']:.1%}"
        )
    else:
        per_pos = "  ".join(
            f"pos{i}={metrics[f'eval/one_step_digit_acc/pos_{i}']:.1%}"
            for i in range(tok.n_digits)
        )
        print(
            f"{checkpoint}\n  dim={dim}  "
            f"one-step (parallel per-digit)={metrics['eval/one_step_acc/mean']:.1%}"
            f"  [{per_pos}]"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate width-sweep checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=SWEEP_CHECKPOINTS,
        help="Local paths or hf:<relpath> into the published dataset",
    )
    parser.add_argument("--n-per-bucket", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        evaluate_checkpoint(checkpoint, n_per_bucket=args.n_per_bucket, seed=args.seed)


if __name__ == "__main__":
    main()
