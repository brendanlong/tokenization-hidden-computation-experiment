"""Analysis for the retok experiment: the four Phase-1 results.

Given a trained checkpoint, computes:

1. **Accuracy split** — CoT vs direct accuracy per carry-chain length
   (also produced by training eval; recomputed here for the figure).
2. **Serial-depth / compute accounting** — the real CoT answer occupies ``D``
   forward positions; the canonical replay occupies ``1``. Combined with the
   measured direct-mode one-step accuracy, the replay implies a capability the
   model demonstrably lacks.
3. **Probe recovery** — a linear probe reads the incoming carry per answer
   position on the *real* CoT stream (state is computed step by step); on the
   canonical replay there is a single answer position, so the per-position
   trajectory has no positions to read.
4. **Calibration gap** — the mean direct-mode probability assigned to the
   correct merged answer vs. the (100%) accuracy of re-tokenized correct CoT
   transcripts: transcripts are far "luckier" than the model's own one-step
   distribution allows — a text-only detector.

Run:
    uv run python -m retok.analysis \
        --checkpoint hf:checkpoints/retok-main-s0/final.pt \
        --out data/retok/analysis
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor

from common.checkpoint import resolve_checkpoint
from common.gpu import resolve_device
from retok.data import (
    FMT_COT,
    FMT_DIRECT,
    compute_arithmetic,
    encode_eval_batch,
    make_eval_buckets,
)
from retok.model import RetokTransformer
from retok.tokenizer import RetokTokenizer
from retok.training import evaluate, load_model


def _prompt_len(n_digits: int) -> int:
    """Index of the first answer token: bos + D a-digits + '+' + D b-digits + '='."""
    return 2 * n_digits + 3


@torch.no_grad()
def _residuals_at(
    model: RetokTransformer,
    input_ids: Tensor,
    positions: list[int],
    device: torch.device,
) -> list[Tensor]:
    """Per-layer residuals gathered at fixed ``positions`` -> list of (B, P, dim)."""
    input_ids = input_ids.to(device)
    _, residuals = model.forward_with_residuals(input_ids)
    pos = torch.tensor(positions, device=device)
    return [r[:, pos, :].float().cpu() for r in residuals]


def _probe_accuracy(features: Tensor, labels: Tensor, seed: int = 0) -> float:
    """Cross-validated-ish linear probe accuracy (80/20 split).

    Returns 0.5-baseline-relative-free raw accuracy. Degenerate single-class
    labels return the majority-class rate (nothing to recover).
    """
    x = features.numpy()
    y = labels.numpy()
    n = len(y)
    if len(set(y.tolist())) < 2:
        # Single-class bucket: a constant predictor is trivially perfect.
        return 1.0
    split = int(n * 0.8)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    tr, te = perm[:split], perm[split:]
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(x[tr], y[tr])
    preds = np.asarray(clf.predict(x[te]))
    return float(np.mean(preds == np.asarray(y[te])))


@dataclass
class CarryProbeResult:
    """Per-position carry-probe accuracy on the real CoT stream vs. replay."""

    n_digits: int
    n_layers: int
    # cot_carry_acc[layer][position] on the real stream
    cot_carry_acc: list[list[float]]
    # replay: probe the single merged answer position for carry_in[i]
    replay_carry_acc: list[list[float]]  # [layer][digit i]
    # majority-class baseline per digit position (carry base rate)
    carry_base_rate: list[float]
    cot_positions: int  # = n_digits (per-digit answer positions)
    replay_positions: int  # = 1 (merged answer collapses the span)


def probe_carry_recovery(
    model: RetokTransformer,
    tok: RetokTokenizer,
    pairs: list[tuple[int, int]],
    device: torch.device,
) -> CarryProbeResult:
    """Train per-(layer, position) linear probes for the incoming carry.

    carry_in[i] is *computed* at the position that predicts answer token i.
    On the real CoT stream that is a distinct position per digit — ``plen+i-1``
    (digit i is emitted at ``plen+i`` and predicted from the token before it).
    On the canonical replay the whole answer is one merged token predicted from
    a *single* position — the ``=`` at ``plen-1`` — so all D carries would have
    to be read there.

    Both sides are probed **pre-emission** (the residual that must *predict* the
    answer token, not one that has already ingested it), so the comparison is
    apples-to-apples: CoT's D dedicated computing-positions vs the replay's one.

    Note: because every carry is a deterministic function of the operands (which
    attention sees at every position), the ``=`` residual retains some carry
    signal too — re-tokenization destroys the *positional structure* of the
    computation (D computing-positions → 1), not the information itself.
    """
    n_digits = tok.n_digits
    plen = _prompt_len(n_digits)
    # carry labels (B, D)
    carries = torch.tensor(
        [list(compute_arithmetic(a, b, n_digits).carry_in) for a, b in pairs],
        dtype=torch.long,
    )
    base_rate = [
        float((carries[:, i] == carries[:, i].mode().values).float().mean())
        for i in range(n_digits)
    ]

    # --- real CoT stream: the position that computes each digit (predicts it) ---
    cot_batch = encode_eval_batch(tok, pairs, FMT_COT)
    cot_positions = [plen + i - 1 for i in range(n_digits)]
    cot_resid = _residuals_at(model, cot_batch["input_ids"], cot_positions, device)
    cot_acc: list[list[float]] = []
    for layer_resid in cot_resid:  # (B, D, dim)
        cot_acc.append(
            [
                _probe_accuracy(layer_resid[:, i, :], carries[:, i])
                for i in range(n_digits)
            ]
        )

    # --- canonical replay == direct format: the single answer-predicting
    # position is the "=" (plen-1), pre-emission — same causal residual on both
    # streams, so this is the honest one-position analog of CoT's D positions.
    direct_batch = encode_eval_batch(tok, pairs, FMT_DIRECT)
    replay_pos = [plen - 1]
    replay_resid = _residuals_at(model, direct_batch["input_ids"], replay_pos, device)
    replay_acc: list[list[float]] = []
    for layer_resid in replay_resid:  # (B, 1, dim)
        replay_acc.append(
            [
                _probe_accuracy(layer_resid[:, 0, :], carries[:, i])
                for i in range(n_digits)
            ]
        )

    return CarryProbeResult(
        n_digits=n_digits,
        n_layers=len(cot_resid),
        cot_carry_acc=cot_acc,
        replay_carry_acc=replay_acc,
        carry_base_rate=base_rate,
        cot_positions=n_digits,
        replay_positions=1,
    )


@dataclass
class CalibrationResult:
    """Calibration gap: model one-step probability vs re-tokenized accuracy."""

    per_chain_direct_p_correct: dict[int, float]
    per_chain_direct_acc: dict[int, float]
    per_chain_cot_acc: dict[int, float]
    retok_transcript_accuracy: float  # correct CoT transcripts re-tokenized => 1.0
    mean_direct_p_correct: float
    # neg-log-likelihood of observing all-correct under the one-step model
    implied_nll_of_observed: float


def calibration_gap(eval_metrics: dict[str, float], n_digits: int) -> CalibrationResult:
    """Turn the training eval metrics into the calibration-gap detector.

    A re-tokenized correct CoT transcript is, by construction, the correct
    merged answer (accuracy 1.0). If an analyst reads that transcript as a
    single-step generation, its probability under the model is the direct-mode
    P(correct). Summed as NLL over a bucket, observing *all* correct is
    astronomically unlikely under the one-step reading.
    """
    p_correct = {
        c: eval_metrics.get(f"eval/direct_p_correct/chain_{c}", 0.0)
        for c in range(n_digits)
    }
    direct_acc = {
        c: eval_metrics.get(f"eval/direct_acc/chain_{c}", 0.0) for c in range(n_digits)
    }
    cot_acc = {
        c: eval_metrics.get(f"eval/cot_acc/chain_{c}", 0.0) for c in range(n_digits)
    }
    ps = [max(p, 1e-12) for p in p_correct.values()]
    mean_p = sum(ps) / len(ps)
    # NLL of observing one correct transcript per chain under the one-step model
    nll = -sum(torch.tensor(ps).log().tolist())
    return CalibrationResult(
        per_chain_direct_p_correct=p_correct,
        per_chain_direct_acc=direct_acc,
        per_chain_cot_acc=cot_acc,
        retok_transcript_accuracy=1.0,
        mean_direct_p_correct=mean_p,
        implied_nll_of_observed=nll,
    )


def run_analysis(
    checkpoint: str,
    out_dir: Path,
    *,
    n_per_bucket: int,
    seed: int,
) -> dict[str, object]:
    device = resolve_device(require_cuda=False)
    ckpt_path = resolve_checkpoint(checkpoint)
    model, model_config = load_model(ckpt_path, device)
    # The checkpoint is self-describing (base/n_digits live in the model config).
    tok = RetokTokenizer(model_config.n_digits, model_config.base)
    assert tok.vocab_size == model_config.vocab_size, (
        f"vocab mismatch: tok={tok.vocab_size} != ckpt={model_config.vocab_size}"
    )
    n_digits = model_config.n_digits
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = make_eval_buckets(tok, n_per_bucket=n_per_bucket, seed=seed)
    from contextlib import nullcontext

    eval_metrics = evaluate(
        model,
        tok,
        buckets,
        device,
        eval_batch_size=1000,
        autocast_ctx=nullcontext(),
    )
    calib = calibration_gap(eval_metrics, n_digits)

    # Probe on the hardest bucket (longest carry chain) pooled with all buckets
    all_pairs = [p for pairs in buckets.values() for p in pairs]
    probe = probe_carry_recovery(model, tok, all_pairs, device)

    results: dict[str, object] = {
        "checkpoint": checkpoint,
        "n_digits": n_digits,
        "eval_metrics": eval_metrics,
        "calibration": asdict(calib),
        "carry_probe": asdict(probe),
    }
    (out_dir / "analysis.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {out_dir / 'analysis.json'}")

    # Console summary
    print("\n=== Accuracy split (CoT vs direct) ===")
    for c in range(n_digits):
        print(
            f"  chain {c}: CoT={calib.per_chain_cot_acc[c]:.1%}  "
            f"direct={calib.per_chain_direct_acc[c]:.1%}  "
            f"direct P(correct)={calib.per_chain_direct_p_correct[c]:.3f}"
        )
    print("\n=== Serial-depth / compute accounting ===")
    print(f"  real CoT answer positions:  {probe.cot_positions}")
    print(f"  canonical replay positions: {probe.replay_positions}")
    print("\n=== Carry probe (best layer, per digit) — real CoT vs replay ===")
    best_cot = [
        max(probe.cot_carry_acc[L][i] for L in range(probe.n_layers))
        for i in range(n_digits)
    ]
    best_replay = [
        max(probe.replay_carry_acc[L][i] for L in range(probe.n_layers))
        for i in range(n_digits)
    ]
    for i in range(n_digits):
        print(
            f"  carry into digit {i}: CoT position={best_cot[i]:.1%}  "
            f"replay(single = position)={best_replay[i]:.1%}  "
            f"base={probe.carry_base_rate[i]:.1%}"
        )
    print("\n=== Calibration gap ===")
    print(f"  mean direct P(correct): {calib.mean_direct_p_correct:.3f}")
    print(f"  re-tokenized transcript accuracy: {calib.retok_transcript_accuracy:.0%}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a trained retok model")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Local path, or hf:<relpath> into the published dataset "
            "(e.g. hf:checkpoints/retok-main-s0/final.pt)."
        ),
    )
    parser.add_argument("--out", default="data/retok/analysis")
    parser.add_argument("--n-per-bucket", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    run_analysis(
        args.checkpoint,
        Path(args.out),
        n_per_bucket=args.n_per_bucket,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
