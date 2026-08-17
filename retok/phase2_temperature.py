"""Why is one model non-canonical and another not? Temperature vs immunity.

Two candidate explanations for a model generating ~0% non-canonically:

- **Structural**: its tokenizer offers no alternative segmentations. (Measurably
  false for Qwen2.5 vs Llama-3.2 — identical segmentation counts per word.)
- **Behavioral/entropy**: the model puts nearly all mass on the canonical
  continuation, so non-canonical tokens are only sampled from the tail.

The second predicts the rate is a **function of sampling temperature** and should
rise steeply with it; the first predicts ~0 at any temperature. This script
sweeps temperature per model and reports the non-canonical rate, which
distinguishes them.

Run:
    uv run python -m retok.phase2_temperature
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.gpu import resolve_device
from retok.phase2_probe import PROMPTS, diff_spans


@torch.no_grad()
def sweep(
    model_name: str,
    temperatures: list[float],
    *,
    n_samples: int,
    max_new_tokens: int,
    seed: int,
) -> None:
    device = resolve_device(require_cuda=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map={"": device.type}, low_cpu_mem_usage=True
    )
    model.eval()
    use_chat = tokenizer.chat_template is not None
    # A fixed prompt set across temperatures (code+multilingual: where spans live)
    prompts = PROMPTS["code"][:4] + PROMPTS["multilingual_latin"][:4]

    print(f"\n===== {model_name} =====")
    print(f"{'temp':>6}{'non-canon':>12}{'gens':>7}{'per-gen':>9}{'per-token':>11}")
    for temp in temperatures:
        torch.manual_seed(seed)
        n_noncanon = 0
        n_total = 0
        noncanon_tok = 0
        total_tok = 0
        for prompt in prompts:
            text = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if use_chat
                else prompt
            )
            enc = tokenizer(text, return_tensors="pt").to(device)
            plen = enc.input_ids.shape[1]
            greedy = temp <= 0.0
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                **(
                    {}
                    if greedy
                    else {
                        "temperature": temp,
                        "top_k": 0,
                        "top_p": 1.0,
                        "repetition_penalty": 1.0,
                    }
                ),
                num_return_sequences=1 if greedy else n_samples,
                pad_token_id=tokenizer.eos_token_id,
            )
            for seq in out:
                gen = seq[plen:].tolist()
                while gen and gen[-1] in (
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                ):
                    gen.pop()
                if not gen:
                    continue
                result = diff_spans(tokenizer, gen)
                if result is None:
                    continue
                _, spans = result
                n_total += 1
                total_tok += len(gen)
                if spans:
                    n_noncanon += 1
                    noncanon_tok += sum(len(s.actual_tokens) for s in spans)
        per_gen = n_noncanon / max(1, n_total)
        per_tok = noncanon_tok / max(1, total_tok)
        print(
            f"{temp:>6.1f}{n_noncanon:>12}{n_total:>7}{per_gen:>8.0%}{per_tok:>10.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature vs non-canonical rate")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        ],
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.0, 0.7, 1.0, 1.5, 2.0]
    )
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for model_name in args.models:
        sweep(
            model_name,
            args.temperatures,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
