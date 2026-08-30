"""Training loop and evaluation for the retok experiment.

Kept separate from the CLI (``train.py``) so the loop and the per-format /
per-difficulty eval are importable by analysis code and tests.
"""

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from common.checkpoint import save_model_checkpoint
from common.schedule import should_log_and_eval
from common.wandb_utils import finish_wandb, init_wandb, log_metrics
from retok.config import RetokModelConfig
from retok.data import (
    FMT_COT,
    FMT_DIRECT,
    encode_eval_batch,
    encode_one_step,
    eq_position,
    make_eval_buckets,
)
from retok.model import RetokTransformer, create_model
from retok.tokenizer import RetokTokenizer


@dataclass
class TrainingResult:
    final_eval: dict[str, float]
    checkpoint_path: Path | None
    total_steps: int


def compute_lm_loss(logits: Tensor, input_ids: Tensor, answer_mask: Tensor) -> Tensor:
    """Causal-shift cross-entropy over the answer tokens only.

    Token at position ``t`` is predicted from ``logits[t-1]``; ``answer_mask``
    marks answer-token positions (answer digits/merged token + eos). Non-answer
    targets (operands, structure) are excluded — they carry no learnable signal
    (operands are random) and would only dilute the metric.
    """
    shift_logits = logits[:, :-1]
    shift_targets = input_ids[:, 1:]
    mask = answer_mask[:, 1:]
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_targets = shift_targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


@torch.no_grad()
def _masked_token_accuracy(
    logits: Tensor, input_ids: Tensor, answer_mask: Tensor
) -> Tensor:
    shift_logits = logits[:, :-1]
    shift_targets = input_ids[:, 1:]
    mask = answer_mask[:, 1:]
    preds = shift_logits.argmax(dim=-1)
    correct = (preds == shift_targets) & mask
    return correct.sum().float() / mask.sum().clamp(min=1).float()


@torch.no_grad()
def _score_format_batch(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    tok: RetokTokenizer,
    fmt: int,
    device: torch.device,
    autocast_ctx: object,
) -> tuple[Tensor, Tensor]:
    """Return (per-example all-tokens-correct, per-example P(correct answer)).

    Accuracy uses argmax **restricted to the format's own token subspace**
    (digits for CoT, merged tokens for direct). This is deliberate: the CoT
    first digit and the direct merged token are both predicted from the ``=``
    position, so a full-vocab argmax conflates the *format choice* (a property
    of the uncued mixture) with *arithmetic correctness* (what we want to
    measure). Restricting to the format's subspace measures "given this format
    path, did the model compute the right answer".

    P(correct answer) stays a **full-softmax** teacher-forced joint probability
    — for direct format it is exactly the unconditional probability the model
    assigns to emitting the correct merged answer at ``=`` (the "lucky sampler"
    calibration quantity, which must account for format-choice mass).
    """
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    positions = batch["answer_token_positions"].to(device, non_blocking=True)
    targets = batch["answer_targets"].to(device, non_blocking=True)
    with autocast_ctx:  # type: ignore[union-attr]
        logits = model(input_ids).float()
    vocab = logits.size(-1)
    pred_idx = positions - 1  # predict token at p from logits[p-1]
    pred_logits = torch.gather(
        logits, 1, pred_idx.unsqueeze(-1).expand(-1, -1, vocab)
    )  # (B, n_answer, V)
    if fmt == FMT_COT:
        lo, hi = tok.digit_offset, tok.digit_offset + tok.base
    else:
        lo, hi = tok.merged_base, tok.merged_base + tok.n_merged
    restricted_pred = pred_logits[..., lo:hi].argmax(dim=-1) + lo
    per_example_correct = (restricted_pred == targets).all(dim=1)
    probs = pred_logits.softmax(dim=-1)  # full vocab
    p_tok = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, n_answer)
    p_answer = p_tok.prod(dim=1)  # teacher-forced joint prob
    return per_example_correct, p_answer


def compute_one_step_loss(
    model: RetokTransformer,
    input_ids: Tensor,
    answer_digits: Tensor,
    eq_pos: int,
    base: int,
) -> Tensor:
    """Parallel per-digit CE for the one-step (no-CoT) control at ``=``."""
    logits = model.one_step_logits(input_ids, eq_pos)  # (B, D, base)
    return F.cross_entropy(logits.reshape(-1, base), answer_digits.reshape(-1))


@torch.no_grad()
def evaluate_one_step(
    model: RetokTransformer,
    tok: RetokTokenizer,
    buckets: dict[int, list[tuple[int, int]]],
    device: torch.device,
    *,
    eval_batch_size: int,
    autocast_ctx: object,
) -> dict[str, float]:
    """One-step capability: parallel per-digit readout at ``=``.

    Reports per-carry-chain-length "all D digits correct" accuracy and, pooled
    across buckets, accuracy per digit *position*, with **no merged-vocab
    retrieval wall** (each head is base-way). Empirically the one-step failure
    is a per-position *arithmetic-capacity* limit, not a carry-chain-depth one:
    at dim=16 accuracy is flat in carry-chain length and the model drops the
    *units* digit (pos 0, which has no incoming carry) while getting the higher
    digits right — the capacity at ``=`` is split D ways, so the hardest exact
    mod-base add is dropped, whereas CoT gives each digit its own position.
    (Addition is TC⁰/parallelizable, so there is no clean depth signature here.)
    """
    model.eval()
    eq_pos = eq_position(tok.n_digits)
    metrics: dict[str, float] = {}
    all_correct = torch.tensor(0.0, device=device)
    total = 0
    pos_correct = torch.zeros(tok.n_digits, device=device)
    pos_total = 0
    for chain_len, pairs in sorted(buckets.items()):
        if not pairs:  # unfillable bucket (rare long chain in a large base)
            continue
        c_correct = torch.tensor(0.0, device=device)
        count = 0
        for i in range(0, len(pairs), eval_batch_size):
            chunk = pairs[i : i + eval_batch_size]
            items = [encode_one_step(tok, a, b) for a, b in chunk]
            input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
            digits = torch.stack([it["answer_digits"] for it in items]).to(device)
            with autocast_ctx:  # type: ignore[union-attr]
                logits = model.one_step_logits(input_ids, eq_pos).float()
            preds = logits.argmax(dim=-1)  # (B, D)
            correct = preds == digits
            c_correct += correct.all(dim=1).sum()
            pos_correct += correct.sum(dim=0)
            count += len(chunk)
            pos_total += len(chunk)
        metrics[f"eval/one_step_acc/chain_{chain_len}"] = (
            c_correct / max(1, count)
        ).item()
        all_correct += c_correct
        total += count
    metrics["eval/one_step_acc/mean"] = (all_correct / max(1, total)).item()
    for i, pc in enumerate((pos_correct / max(1, pos_total)).tolist()):
        metrics[f"eval/one_step_digit_acc/pos_{i}"] = pc
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tok: RetokTokenizer,
    buckets: dict[int, list[tuple[int, int]]],
    device: torch.device,
    *,
    eval_batch_size: int,
    autocast_ctx: object,
    merged_operands: bool = False,
) -> dict[str, float]:
    """Per-format, per-carry-chain-length accuracy on held-out buckets.

    Reports CoT and direct accuracy for each chain length, and the mean
    direct-mode probability assigned to the correct merged answer (the
    calibration quantity the "lucky sampler" story rests on).
    """
    model.eval()
    metrics: dict[str, float] = {}
    cot_correct_all = torch.tensor(0.0, device=device)
    cot_total = 0
    direct_correct_all = torch.tensor(0.0, device=device)
    direct_total = 0
    for chain_len, pairs in sorted(buckets.items()):
        if not pairs:
            continue
        for fmt, tag in ((FMT_COT, "cot"), (FMT_DIRECT, "direct")):
            correct_sum = torch.tensor(0.0, device=device)
            p_sum = torch.tensor(0.0, device=device)
            count = 0
            for i in range(0, len(pairs), eval_batch_size):
                chunk = pairs[i : i + eval_batch_size]
                batch = encode_eval_batch(
                    tok, chunk, fmt, merged_operands=merged_operands
                )
                correct, p_answer = _score_format_batch(
                    model, batch, tok, fmt, device, autocast_ctx
                )
                correct_sum += correct.sum()
                p_sum += p_answer.sum()
                count += len(chunk)
            acc = (correct_sum / max(1, count)).item()
            metrics[f"eval/{tag}_acc/chain_{chain_len}"] = acc
            if tag == "direct":
                metrics[f"eval/direct_p_correct/chain_{chain_len}"] = (
                    p_sum / max(1, count)
                ).item()
                direct_correct_all += correct_sum
                direct_total += count
            else:
                cot_correct_all += correct_sum
                cot_total += count
    metrics["eval/cot_acc/mean"] = (cot_correct_all / max(1, cot_total)).item()
    metrics["eval/direct_acc/mean"] = (direct_correct_all / max(1, direct_total)).item()
    metrics["eval/acc_gap/mean"] = (
        metrics["eval/cot_acc/mean"] - metrics["eval/direct_acc/mean"]
    )
    return metrics


def load_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[RetokTransformer, RetokModelConfig]:
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location=device)
    config = RetokModelConfig(**ckpt["model_config"])
    model = create_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), config


def train_retok_model(
    *,
    model: RetokTransformer,
    train_loader: DataLoader[dict[str, Tensor]],
    tok: RetokTokenizer,
    model_config: RetokModelConfig,
    device: torch.device,
    mode: str,
    total_steps: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    lr_schedule: str,
    use_compile: bool,
    n_eval_per_bucket: int,
    eval_batch_size: int,
    log_every_steps: int,
    eval_every_steps: int,
    seed: int,
    run_name: str,
    checkpoint_dir: Path,
    use_wandb: bool,
    wandb_project: str,
    wandb_config: dict[str, object],
    merged_operands: bool = False,
) -> TrainingResult:
    print(
        f"Model params: {model.count_parameters():,} "
        f"(vocab={model_config.vocab_size}, {model_config.n_layers}L "
        f"{model_config.dim}d {model_config.n_heads}h)"
    )
    model = model.to(device)
    if use_compile and device.type == "cuda":
        model = torch.compile(model)  # type: ignore[assignment]

    autocast_ctx: object = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = min(warmup_steps, max(1, total_steps // 2))
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1 / warmup, total_iters=warmup
    )
    if lr_schedule == "cosine":
        decay_sched: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup)
            )
        )
    else:
        decay_sched = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=total_steps
        )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup]
    )

    eval_buckets = make_eval_buckets(tok, n_per_bucket=n_eval_per_bucket, seed=seed)

    init_wandb(
        enabled=use_wandb, project=wandb_project, run_name=run_name, config=wandb_config
    )

    eq_pos = eq_position(tok.n_digits)

    def run_eval() -> dict[str, float]:
        m = (
            evaluate_one_step(
                model,
                tok,
                eval_buckets,
                device,
                eval_batch_size=eval_batch_size,
                autocast_ctx=autocast_ctx,
            )
            if mode == "one_step"
            else evaluate(
                model,
                tok,
                eval_buckets,
                device,
                eval_batch_size=eval_batch_size,
                autocast_ctx=autocast_ctx,
                merged_operands=merged_operands,
            )
        )
        model.train()
        return m

    running_loss = torch.tensor(0.0, device=device)
    running_acc = torch.tensor(0.0, device=device)
    running_count = 0
    global_step = 0
    t0 = time.time()
    last_ckpt: Path | None = None
    final_eval: dict[str, float] = {}

    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        # Bound only on the cot path; token accuracy is likewise cot-only.
        logits: Tensor | None = None
        answer_mask: Tensor | None = None
        with autocast_ctx:  # type: ignore[union-attr]
            if mode == "one_step":
                answer_digits = batch["answer_digits"].to(device, non_blocking=True)
                loss = compute_one_step_loss(
                    model, input_ids, answer_digits, eq_pos, tok.base
                )
            else:
                mask = batch["answer_mask"].to(device, non_blocking=True)
                lm_logits = model(input_ids)
                loss = compute_lm_loss(lm_logits, input_ids, mask)
                logits, answer_mask = lm_logits, mask
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        running_loss += loss.detach()
        if logits is not None and answer_mask is not None:
            running_acc += _masked_token_accuracy(
                logits.detach(), input_ids, answer_mask
            )
        running_count += 1

        do_log, do_eval = should_log_and_eval(
            global_step,
            log_every_steps=log_every_steps,
            eval_every_steps=eval_every_steps,
        )
        if do_log:
            avg_loss = (running_loss / running_count).item()
            avg_acc = (running_acc / running_count).item()
            steps_per_sec = running_count / (time.time() - t0)
            log_dict = {
                "train/loss": avg_loss,
                "perf/steps_per_sec": steps_per_sec,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if mode != "one_step":
                log_dict["train/token_acc"] = avg_acc
            if do_eval:
                eval_metrics = run_eval()
                log_dict.update(eval_metrics)
                final_eval = eval_metrics
                if mode == "one_step":
                    summary = f"one_step={eval_metrics['eval/one_step_acc/mean']:.1%}"
                else:
                    summary = (
                        f"cot={eval_metrics['eval/cot_acc/mean']:.1%} "
                        f"direct={eval_metrics['eval/direct_acc/mean']:.1%} "
                        f"gap={eval_metrics['eval/acc_gap/mean']:.1%}"
                    )
                print(
                    f"  [{run_name}] step {global_step:>6}/{total_steps} "
                    f"loss={avg_loss:.4f} {summary} | {steps_per_sec:.1f} it/s",
                    flush=True,
                )
            else:
                print(
                    f"  [{run_name}] step {global_step:>6}/{total_steps} "
                    f"loss={avg_loss:.4f} | {steps_per_sec:.1f} it/s",
                    flush=True,
                )
            log_metrics(log_dict, step=global_step, enabled=use_wandb)
            running_loss = torch.tensor(0.0, device=device)
            running_acc = torch.tensor(0.0, device=device)
            running_count = 0
            t0 = time.time()

    # Final eval + checkpoint
    final_eval = run_eval()
    last_ckpt = save_model_checkpoint(
        model,
        global_step,
        model_config.model_dump(),
        checkpoint_dir,
        filename="final.pt",
    )
    # Metadata sidecar, mirroring the layout of the published checkpoints in the
    # HuggingFace dataset (see common/artifacts.py).
    last_ckpt.with_suffix(".pt.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "mode": mode,
                "total_steps": global_step,
                **final_eval,
            },
            indent=2,
            default=str,
        )
    )
    log_metrics(final_eval, step=global_step, enabled=use_wandb)
    finish_wandb(enabled=use_wandb)
    return TrainingResult(
        final_eval=final_eval, checkpoint_path=last_ckpt, total_steps=global_step
    )
