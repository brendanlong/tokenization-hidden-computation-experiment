"""Sampled (temperature-1) format mix and text-level accuracy for the toy models.

Draws one temperature-1 sample per problem, classifies each output by the
*observed* format of its answer tokens (all-digit / single merged / other),
and scores correctness on the DECODED TEXT — the answer string — not on
token identity, so a correct answer counts regardless of which tokens
carried it. Prompts can be encoded either way (`--prompt-format`), which is
the row axis of the writeup's behaviour grid.

    uv run python -m retok.sample_eval                          # retok-main seeds
    uv run python -m retok.sample_eval \
        --checkpoints data/retok/checkpoints/retok-mergedops-s0/final.pt \
        --prompt-format merged

CPU is fine (~25k-parameter model). Reference output for the retok-main
seeds (digit prompts): digit-by-digit 71.0-71.5% of outputs at 98.0-99.2%
text accuracy; merged 28.4-28.8% at 0.69-1.05%; other <=0.1%.
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


def sample_eval(
    checkpoint: str,
    n_pairs: int,
    pair_seed: int,
    torch_seed: int,
    prompt_format: str = "digit",
) -> dict:
    tok = RetokTokenizer(n_digits=3, base=10)
    rng = random.Random(pair_seed)
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.randrange(0, 1000), rng.randrange(0, 1000)
        if a + b < 1000:  # no overflow: every answer is exactly one merged token
            pairs.append((a, b))

    prompts, answers = [], []
    for a, b in pairs:
        if prompt_format == "digit":
            prompt = encode_example(tok, a, b, fmt=1).input_ids[:PROMPT_LEN]
        else:
            prompt = [
                tok.bos_id,
                tok.merged_token(a),
                tok.plus_id,
                tok.merged_token(b),
                tok.eq_id,
            ]
        prompts.append(prompt)
        # correctness is TEXT-level: the reversed answer string, however tokenized
        answers.append("".join(str(d) for d in tok.rev_digits(a + b)))

    path = (
        resolve_checkpoint(checkpoint) if checkpoint.startswith("hf:") else checkpoint
    )
    ck = torch.load(path, weights_only=True)
    model = create_model(RetokModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    torch.manual_seed(torch_seed)

    st = {
        "digit": 0,
        "merged": 0,
        "other": 0,
        "digit_ok": 0,
        "merged_ok": 0,
        "other_ok": 0,
    }
    plen = len(prompts[0])
    with torch.no_grad():
        batch = 500
        for i in range(0, len(pairs), batch):
            chunk = prompts[i : i + batch]
            width = max(len(p) for p in chunk)
            seq = torch.full((len(chunk), width), tok.pad_id, dtype=torch.long)
            for j, pr in enumerate(chunk):
                seq[j, : len(pr)] = torch.tensor(pr)
            for _ in range(4):  # 3 answer digits + eos is the longest answer
                probs = torch.softmax(model(seq)[:, -1, :], dim=-1)
                seq = torch.cat([seq, torch.multinomial(probs, 1)], dim=1)
            for j in range(seq.shape[0]):
                out = seq[j, plen:].tolist()
                if tok.eos_id in out:
                    out = out[: out.index(tok.eos_id)]
                # observed format of the emitted answer tokens
                if out and all(tok.is_digit_token(t) for t in out):
                    fmt = "digit"
                elif len(out) == 1 and tok.is_merged_token(out[0]):
                    fmt = "merged"
                else:
                    fmt = "other"
                # text-level correctness: decoded surface == answer string
                surface = "".join(tok.token_to_str(t) for t in out)
                st[fmt] += 1
                st[f"{fmt}_ok"] += surface == answers[i + j]
    return st


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoints",
        nargs="+",
        default=[f"hf:checkpoints/retok-main-s{s}/final.pt" for s in (0, 1, 2)],
        help="local paths or hf:checkpoints/<run>/final.pt",
    )
    p.add_argument("--prompt-format", choices=["digit", "merged"], default="digit")
    p.add_argument("--n-pairs", type=int, default=2000)
    p.add_argument("--pair-seed", type=int, default=123)
    p.add_argument("--torch-seed", type=int, default=0)
    a = p.parse_args()

    for ckpt in a.checkpoints:
        st = sample_eval(ckpt, a.n_pairs, a.pair_seed, a.torch_seed, a.prompt_format)
        n = st["digit"] + st["merged"] + st["other"]
        ok = st["digit_ok"] + st["merged_ok"] + st["other_ok"]
        name = ckpt.removeprefix("hf:checkpoints/").removesuffix("/final.pt")
        print(
            f"{name} [{a.prompt_format} prompts]:"
            f" digit-by-digit {st['digit'] / n:5.1%}"
            f" (text-acc {st['digit_ok'] / max(1, st['digit']):6.1%})"
            f"   merged {st['merged'] / n:5.1%}"
            f" (text-acc {st['merged_ok'] / max(1, st['merged']):5.2%})"
            f"   other {st['other'] / n:4.1%}"
            f"   overall {ok / n:5.1%}"
        )


if __name__ == "__main__":
    main()
