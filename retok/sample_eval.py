"""Sampled (temperature-1) format mix and accuracy for the toy checkpoints.

The training eval reports per-format *argmax* accuracies (best case within
each branch). This answers the plainer question — what does the model
actually do when you sample? — by drawing one temperature-1 sample per
problem from the published checkpoints and classifying each output by the
format of its first answer token.

    uv run python -m retok.sample_eval            # all 3 seeds, 2000 problems
    uv run python -m retok.sample_eval --runs retok-main-s0 --n-pairs 500

CPU is fine (~25k-parameter model). Reference output (2026-08-30, seed 123
problems, torch seed 0): digit-by-digit 71.0-71.5% of outputs at 98.0-99.2%
accuracy; merged 28.4-28.8% at 0.69-1.05%; malformed <=0.1%; overall
70.4-71.0%.
"""

from __future__ import annotations

import argparse
import random

import torch

from common.checkpoint import resolve_checkpoint
from retok.config import RetokModelConfig
from retok.data import encode_example
from retok.model import create_model
from retok.tokenizer import RetokTokenizer

PROMPT_LEN = 9  # <bos> a0 a1 a2 + b0 b1 b2 =


def sample_eval(run: str, n_pairs: int, pair_seed: int, torch_seed: int) -> dict:
    tok = RetokTokenizer(n_digits=3, base=10)
    rng = random.Random(pair_seed)
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.randrange(0, 1000), rng.randrange(0, 1000)
        if a + b < 1000:  # no overflow: every answer is exactly one merged token
            pairs.append((a, b))

    prompts, cot_t, mrg_t = [], [], []
    for a, b in pairs:
        e0 = encode_example(tok, a, b, fmt=0)  # merged answer
        e1 = encode_example(tok, a, b, fmt=1)  # digit-by-digit answer
        prompts.append(e1.input_ids[:PROMPT_LEN])
        mrg_t.append(e0.input_ids[e0.answer_token_positions[0]])
        p1 = e1.answer_token_positions[0]
        cot_t.append(e1.input_ids[p1 : p1 + 3])

    ck = torch.load(
        resolve_checkpoint(f"hf:checkpoints/{run}/final.pt"), weights_only=True
    )
    model = create_model(RetokModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    torch.manual_seed(torch_seed)

    st = {"digit": 0, "merged": 0, "other": 0, "digit_ok": 0, "merged_ok": 0}
    with torch.no_grad():
        batch = 500
        for i in range(0, len(pairs), batch):
            seq = torch.tensor(prompts[i : i + batch])
            for _ in range(4):  # 3 answer digits + eos is the longest answer
                probs = torch.softmax(model(seq)[:, -1, :], dim=-1)
                seq = torch.cat([seq, torch.multinomial(probs, 1)], dim=1)
            for j in range(seq.shape[0]):
                out = seq[j, PROMPT_LEN:].tolist()
                k = i + j
                if tok.is_digit_token(out[0]):
                    st["digit"] += 1
                    st["digit_ok"] += out[:3] == cot_t[k]
                elif tok.is_merged_token(out[0]):
                    st["merged"] += 1
                    st["merged_ok"] += out[0] == mrg_t[k]
                else:
                    st["other"] += 1
    return st


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="+",
        default=["retok-main-s0", "retok-main-s1", "retok-main-s2"],
    )
    p.add_argument("--n-pairs", type=int, default=2000)
    p.add_argument("--pair-seed", type=int, default=123)
    p.add_argument("--torch-seed", type=int, default=0)
    a = p.parse_args()

    for run in a.runs:
        st = sample_eval(run, a.n_pairs, a.pair_seed, a.torch_seed)
        n = st["digit"] + st["merged"] + st["other"]
        print(
            f"{run}: digit-by-digit {st['digit'] / n:5.1%}"
            f" (acc {st['digit_ok'] / max(1, st['digit']):6.1%})"
            f"   merged {st['merged'] / n:5.1%}"
            f" (acc {st['merged_ok'] / max(1, st['merged']):5.2%})"
            f"   other {st['other'] / n:4.1%}"
            f"   overall acc {(st['digit_ok'] + st['merged_ok']) / n:5.1%}"
        )


if __name__ == "__main__":
    main()
