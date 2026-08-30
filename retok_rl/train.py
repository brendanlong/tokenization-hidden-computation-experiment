"""GRPO on decimal expansion, rewarding correct leading digits of DECODED text.

    uv run python -m retok_rl.train --steps 3000

Every tokenization that decodes to the right digits scores identically, so
segmentation is a free degree of freedom the reward cannot see. The question is
whether the policy moves off canonical anyway. See EXPERIMENT_PLAN.md.

TRL's GRPO defaults are already the configuration we want -- temperature 1.0,
top_k 0, top_p 1.0, beta 0.0 -- so arm A is stock GRPO rather than a tuned
setup. Arm B sets --beta > 0 to ask whether a KL term suppresses the behaviour
once the task is mastered.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer  # type: ignore[attr-defined]

from common.gpu import resolve_device
from common.wandb_utils import finish_wandb, init_wandb, log_metrics
from retok_rl.data import build, divisor_class, split_divisors
from retok_rl.metrics import (
    digit_prefix,
    leading_correct,
    summarise,
    tokens_covering_digits,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from retok_rl.data import Example

Rollout = tuple[list[int], str]


def make_reward(places: int) -> Callable[..., list[float]]:
    """Reward = count of correct leading digits, on the DECODED completion.

    Absolute count, not a fraction: a length-normalised reward would make
    "emit one correct digit and stop" score 1.0, which the policy finds at once.
    """

    def reward(completions: list[str], target: list[str], **_: object) -> list[float]:
        return [
            float(leading_correct(digit_prefix(c), t))
            for c, t in zip(completions, target, strict=True)
        ]

    return reward


class CollapseGuard(TrainerCallback):
    """Stop when every GRPO group has identical rewards, i.e. zero advantage.

    Arm A collapsed at step ~250 -- entropy fell 4.65 -> 0.004 nats and
    frac_reward_zero_std pinned at 1.0 -- and the remaining 2750 steps were
    exact no-ops that still cost GPU time. Once reward variance is zero within
    every group there is no gradient, so there is nothing to wait for.
    """

    def __init__(self, patience: int = 20) -> None:
        self.patience = patience
        self.streak = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        if not logs:
            return
        frac = logs.get("frac_reward_zero_std")
        if frac is None:
            return
        self.streak = self.streak + 1 if frac >= 1.0 else 0
        if self.streak >= self.patience:
            print(
                f">>> ABORT: frac_reward_zero_std == 1.0 for {self.streak} logs "
                f"(entropy {logs.get('entropy', float('nan')):.4f}). "
                "No gradient remains; stopping instead of burning steps.",
                flush=True,
            )
            control.should_training_stop = True


class TokenizationEval(TrainerCallback):
    """Log the curves that matter: attractor mix, single-digit fraction, held-out.

    Runs its own generation rather than reading TRL internals, so the sampling
    config is explicit and the token IDs (not just decoded text) are available --
    the whole point is measuring segmentation, which decoded text destroys.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tok: PreTrainedTokenizerBase,
        train_ex: list[Example],
        held_ex: list[Example],
        places: int,
        every: int,
        n_samples: int,
        use_wandb: bool,
        jsonl_path: Path | None = None,
        run_name: str = "",
        seed: int = 0,
        n_eval_prompts: int = 24,
    ) -> None:
        self.model, self.tok = model, tok
        self.train_ex, self.held_ex = train_ex, held_ex
        self.places, self.every, self.n = places, every, n_samples
        self.use_wandb = use_wandb
        # Per-rollout artifact retention (ported from the reversal arm after
        # Run 4 shipped with training-log metrics only). Stamped with run
        # identity: the file is opened in append mode.
        self.jsonl_path = jsonl_path
        self.run_name, self.seed = run_name, seed
        self.n_eval_prompts = n_eval_prompts

    @torch.no_grad()
    def _rollouts(
        self, examples: list[Example], split: str, step: int
    ) -> tuple[list[Rollout], dict[str, list[Rollout]]]:
        self.model.eval()
        out: list[Rollout] = []
        by_class: dict[str, list[Rollout]] = defaultdict(list)
        records = []
        for ex in examples:
            # Build the tensor explicitly: BatchEncoding's typing confuses the
            # checker, and this is unambiguous about what reaches generate().
            ids = torch.tensor([self.tok.encode(ex.prompt)], device=self.model.device)
            gen = self.model.generate(  # type: ignore[operator]
                input_ids=ids,
                max_new_tokens=self.places + 8,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                num_return_sequences=self.n,
                pad_token_id=self.tok.eos_token_id,
            )
            for seq in gen:
                out_ids = seq[ids.shape[1] :].tolist()
                kept, digits = tokens_covering_digits(self.tok, out_ids)
                out.append((out_ids, ex.target))
                by_class[divisor_class(ex.b)].append((out_ids, ex.target))
                records.append(
                    {
                        "run": self.run_name,
                        "seed": self.seed,
                        "step": step,
                        "split": split,
                        "b": ex.b,
                        "target": ex.target,
                        "completion_ids": out_ids,
                        "kept_ids": kept,
                        "digits": digits,
                        "leading_correct": leading_correct(digits, ex.target),
                    }
                )
        if self.jsonl_path is not None:
            with self.jsonl_path.open("a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        self.model.train()
        return out, by_class

    def run_eval(self, step: int) -> None:
        # Subsample train prompts: eval is ~10x the cost of a training step,
        # and the train-side estimate does not need all 77 divisors every time.
        sample = self.train_ex[:: max(1, len(self.train_ex) // self.n_eval_prompts)]
        roll, by_class = self._rollouts(sample, "train", step)
        metrics = {f"train/{k}": v for k, v in summarise(self.tok, roll).items()}
        for cls, rs in by_class.items():
            s = summarise(self.tok, rs)
            metrics[f"class/{cls}/frac_single_digit"] = s["frac_single_digit_tokens"]
            metrics[f"class/{cls}/mean_reward"] = s["mean_reward"]
        held, _ = self._rollouts(self.held_ex, "heldout", step)
        metrics.update(
            {f"heldout/{k}": v for k, v in summarise(self.tok, held).items()}
        )
        log_metrics(metrics, step=step, enabled=self.use_wandb)
        print(
            f"  step {step:>5}  reward {metrics['train/mean_reward']:6.2f}"
            f"  single-digit {metrics['train/frac_single_digit_tokens']:6.1%}"
            f"  canon {metrics['train/attractor/canonical']:5.1%}"
            f"  greedy {metrics['train/attractor/greedy-longest']:5.1%}"
            f"  nc-tok {metrics['train/roundtrip/tok_non_canonical']:5.2%}"
            f"  cr-single {metrics['train/correct_region/frac_single_digit']:5.1%}"
            f"  held-out reward {metrics['heldout/mean_reward']:5.2f}",
            flush=True,
        )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        # Step-0 baseline (ported from the reversal arm): the reference every
        # drift claim is measured against.
        self.run_eval(0)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if state.global_step % self.every or state.global_step == 0:
            return
        self.run_eval(state.global_step)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--places", type=int, default=30)
    p.add_argument("--n-held-out", type=int, default=20)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient (arm B: >0)")
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--use-adaptive-entropy", action="store_true")
    p.add_argument("--entropy-target", type=float, default=2.0)
    p.add_argument("--collapse-patience", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/retok_rl/run")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="retok_rl")
    p.add_argument("--wandb-run-name", default=None)
    a = p.parse_args()

    device = resolve_device()
    print(f"Device: {device}")
    torch.manual_seed(a.seed)

    train_b, held_b = split_divisors(a.n_held_out, a.seed)
    train_ex, held_ex = build(train_b, a.places), build(held_b, a.places)
    print(f"divisors: {len(train_b)} train, {len(held_b)} held out -> {held_b}")

    # Pass the token explicitly. Relying on the HF_TOKEN environment variable
    # is not enough on some images: huggingface_hub prefers a stored
    # ~/.cache/huggingface/token when one exists, so a correct env token gets
    # ignored and gated repos 401 even though the token is present (confirmed:
    # the pod reported HF_TOKEN=37 chars and still failed).
    hf_token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(a.model, token=hf_token)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, token=hf_token)
    model.to(device)  # type: ignore[arg-type]

    ds = Dataset.from_list(
        [{"prompt": e.prompt, "target": e.target, "b": e.b} for e in train_ex]
    )

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rollouts.jsonl"

    run_name = a.wandb_run_name or f"retok_rl_beta{a.beta}_{a.places}p"
    init_wandb(
        enabled=not a.no_wandb,
        project=a.wandb_project,
        run_name=run_name,
        config={k: v for k, v in vars(a).items()},
    )

    cfg = GRPOConfig(
        output_dir=a.output_dir,
        max_steps=a.steps,
        per_device_train_batch_size=a.batch_size,
        num_generations=a.num_generations,
        max_completion_length=a.places + 8,
        learning_rate=a.lr,
        beta=a.beta,  # 0.0 = arm A, no KL to reference
        entropy_coef=a.entropy_coef,
        use_adaptive_entropy=a.use_adaptive_entropy,
        entropy_target=a.entropy_target,
        temperature=1.0,
        top_k=0,
        top_p=1.0,  # pure sampling: on-policy
        logging_steps=10,
        save_strategy="no",
        report_to=[],  # we log through common.wandb_utils
        seed=a.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward(a.places),  # type: ignore[arg-type]
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[
            TokenizationEval(
                model,
                tok,
                train_ex,
                held_ex,
                a.places,
                a.eval_every,
                a.eval_samples,
                not a.no_wandb,
                jsonl_path,
                run_name,
                a.seed,
            )
        ],
    )
    trainer.train()
    finish_wandb(enabled=not a.no_wandb)
    print(f"rollout artifacts: {jsonl_path}")


if __name__ == "__main__":
    main()
