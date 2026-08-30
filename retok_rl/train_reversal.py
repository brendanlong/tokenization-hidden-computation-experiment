"""Arm C: GRPO on word reversal, rewarding correct leading letters of DECODED text.

    uv run python -m retok_rl.train_reversal --steps 2000

Reversal is the compute-aligned task (output order = computation order), so
the pre-registered prediction is drift toward the all-single-char attractor —
the opposite weighting from the expansion arm. Reward reads the decoded
completion only; segmentation stays invisible to it. See EXPERIMENT_PLAN.md,
Arm C.

Differences from the expansion arm's harness, both deliberate:
- a step-0 baseline eval runs before any training, so drift is measured
  against the un-trained policy rather than the first logged step;
- every eval appends per-rollout records (token IDs, target, attractor,
  correctness) to ``rollouts.jsonl`` so this arm is independently
  recomputable, unlike the expansion runs.
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
from retok_rl.metrics import leading_correct
from retok_rl.reversal import (
    Example,
    Rollout,
    build_reversal,
    classify_letters,
    letter_prefix,
    split_words,
    summarise_reversal,
    tokens_covering_letters,
)
from retok_rl.train import CollapseGuard

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_WORD_LEN = 8


def make_reward() -> Callable[..., list[float]]:
    """Reward = count of correct leading letters, on the DECODED completion.

    Absolute count, not a fraction — same anti-hack rationale as the
    expansion arm: a normalised reward makes "one correct letter and stop"
    score 1.0. (Run 5 lesson: absolute count is still hackable from the
    other side — with nothing penalising trailing garbage, the policy
    converged on a correct 2-3 char prefix plus junk. A follow-up arm
    should add an exactness/termination term.)

    Definitional note: this uses ``letter_prefix`` on the full decoded
    completion, while the eval callback's letter run comes from per-token
    decoding (``tokens_covering_letters``), which truncates at U+FFFD when
    a multi-byte character spans tokens. The two disagree on ~2% of
    rollouts; measured mean leading-correct is identical to 4 decimals.
    """

    def reward(completions: list[str], target: list[str], **_: object) -> list[float]:
        return [
            float(leading_correct(letter_prefix(c), t))
            for c, t in zip(completions, target, strict=True)
        ]

    return reward


class ReversalEval(TrainerCallback):
    """Correctness + effort + attractor curves, per length class, with artifacts.

    Runs its own generation (explicit sampling config, token IDs retained) and
    appends one JSONL record per rollout at every eval, so the run is
    recomputable from its artifacts.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tok: PreTrainedTokenizerBase,
        train_ex: list[Example],
        held_ex: list[Example],
        every: int,
        n_samples: int,
        use_wandb: bool,
        jsonl_path: Path,
        run_name: str,
        seed: int,
        n_eval_prompts: int = 36,
    ) -> None:
        self.model, self.tok = model, tok
        self.train_ex, self.held_ex = train_ex, held_ex
        self.every, self.n = every, n_samples
        self.use_wandb = use_wandb
        self.jsonl_path = jsonl_path
        # Stamped into every record: the file is opened in append mode, so
        # without run identity a smoke run and a real run on the same output
        # dir would silently concatenate and collide at step 0.
        self.run_name, self.seed = run_name, seed
        self.n_eval_prompts = n_eval_prompts

    @torch.no_grad()
    def _rollouts(
        self, examples: list[Example], split: str, step: int
    ) -> tuple[list[Rollout], dict[int, list[Rollout]]]:
        self.model.eval()
        out: list[Rollout] = []
        by_len: dict[int, list[Rollout]] = defaultdict(list)
        records = []
        for ex in examples:
            ids = torch.tensor([self.tok.encode(ex.prompt)], device=self.model.device)
            gen = self.model.generate(  # type: ignore[operator]
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=MAX_WORD_LEN + 8,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                num_return_sequences=self.n,
                pad_token_id=self.tok.eos_token_id,
            )
            for seq in gen:
                out_ids = seq[ids.shape[1] :].tolist()
                kept, produced = tokens_covering_letters(self.tok, out_ids)
                out.append((kept, ex.target))
                by_len[len(ex.word)].append((kept, ex.target))
                records.append(
                    {
                        "run": self.run_name,
                        "seed": self.seed,
                        "step": step,
                        "split": split,
                        "word": ex.word,
                        "target": ex.target,
                        "completion_ids": out_ids,
                        "kept_ids": kept,
                        "produced": produced,
                        "attractor": classify_letters(self.tok, kept, produced)
                        if kept
                        else "empty",
                        "leading_correct": leading_correct(produced, ex.target),
                    }
                )
        with self.jsonl_path.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        self.model.train()
        return out, by_len

    def run_eval(self, step: int) -> None:
        sample = self.train_ex[:: max(1, len(self.train_ex) // self.n_eval_prompts)]
        roll, by_len = self._rollouts(sample, "train", step)
        metrics = {
            f"train/{k}": v for k, v in summarise_reversal(self.tok, roll).items()
        }
        for length, rs in by_len.items():
            s = summarise_reversal(self.tok, rs)
            metrics[f"len{length}/mean_reward"] = s["mean_reward"]
            metrics[f"len{length}/frac_single_char"] = s["frac_single_char_tokens"]
        held, _ = self._rollouts(self.held_ex, "heldout", step)
        metrics.update(
            {f"heldout/{k}": v for k, v in summarise_reversal(self.tok, held).items()}
        )
        log_metrics(metrics, step=step, enabled=self.use_wandb)
        print(
            f"  step {step:>5}  reward {metrics['train/mean_reward']:5.2f}"
            f"  exact {metrics['train/exact_match']:5.1%}"
            f"  attempted {metrics['train/attempted']:5.1%}"
            f"  single-char {metrics['train/frac_single_char_tokens']:6.1%}"
            f"  canon {metrics['train/attractor/canonical']:5.1%}"
            f"  greedy {metrics['train/attractor/greedy-longest']:5.1%}"
            f"  held reward {metrics['heldout/mean_reward']:5.2f}"
            f"  held single-char {metrics['heldout/frac_single_char_tokens']:6.1%}",
            flush=True,
        )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        # The step-0 baseline every drift claim is measured against.
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
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n-held-out", type=int, default=60)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--collapse-patience", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/retok_rl/reversal")
    p.add_argument("--bf16", action="store_true", help="load + train in bf16")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="retok_rl")
    p.add_argument("--wandb-run-name", default=None)
    a = p.parse_args()

    device = resolve_device()
    print(f"Device: {device}")
    torch.manual_seed(a.seed)

    hf_token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(a.model, token=hf_token)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, token=hf_token, dtype=torch.bfloat16 if a.bf16 else None
    )
    model.to(device)  # type: ignore[arg-type]

    train_w, held_w = split_words(a.n_held_out, a.seed)
    train_ex, held_ex = build_reversal(train_w, tok), build_reversal(held_w, tok)
    print(f"words: {len(train_w)} train, {len(held_w)} held out")

    ds = Dataset.from_list(
        [{"prompt": e.prompt, "target": e.target, "word": e.word} for e in train_ex]
    )

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rollouts.jsonl"

    run_name = a.wandb_run_name or f"retok-rl-reversal-s{a.seed}"
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
        max_completion_length=MAX_WORD_LEN + 8,
        learning_rate=a.lr,
        beta=a.beta,
        entropy_coef=a.entropy_coef,
        temperature=1.0,
        top_k=0,
        top_p=1.0,  # pure sampling: on-policy
        bf16=a.bf16,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=a.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward(),  # type: ignore[arg-type]
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[
            ReversalEval(
                model,
                tok,
                train_ex,
                held_ex,
                a.eval_every,
                a.eval_samples,
                not a.no_wandb,
                jsonl_path,
                run_name,
                a.seed,
            ),
            CollapseGuard(patience=a.collapse_patience),
        ],
    )
    trainer.train()
    finish_wandb(enabled=not a.no_wandb)
    print(f"rollout artifacts: {jsonl_path}")


if __name__ == "__main__":
    main()
