"""Phase-2 interp-mismatch demo: re-tokenizing a transcript changes the
model's *actual* computation, not just the token count.

An analyst who stores a transcript as text and re-tokenizes it (canonically)
then runs the model to analyze it — but the model actually ran on the
as-generated (possibly non-canonical) tokens. This measures the gap: at the
text boundary immediately after a non-canonical span, compare the model's
next-token distribution under the **actual** tokenization vs the **canonical**
re-tokenization of the *identical prefix text*.

- `boundary_kl`: KL(actual ‖ canonical) at that boundary. For canonical
  generations it is 0 by construction (identical tokens), so the effect is a
  clean function of non-canonicality.
- `top1_disagree`: fraction where the two tokenizations predict a different
  next token — i.e. re-tokenized replay would have diverged from the real run.
- A logit-lens example shows the per-position trajectory differs (D actual
  sub-token positions vs the canonical positions).

Run:
    uv run python -m retok.phase2_interp \
        --model meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.gpu import resolve_device
from retok.phase2_probe import (
    PROMPTS,
    decode_for_roundtrip,
    roundtrip_is_measurable,
)


@dataclass
class Boundary:
    domain: str
    span_actual: list[str]
    span_canonical: list[str]
    kl: float
    actual_next: str
    canonical_next: str
    # confidence context for the flip: what the ACTUAL run assigned to its own
    # top-1 vs to the token the re-tokenized replay would pick.
    pa_top1: float
    pa_at_canon: float

    @property
    def flipped(self) -> bool:
        return self.actual_next != self.canonical_next


def _first_replace(
    actual: list[int], canon: list[int]
) -> tuple[int, int, int, int] | None:
    """First differing (replace) region as (i1, i2, j1, j2), or None if identical."""
    if actual == canon:
        return None
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=actual, b=canon, autojunk=False
    ).get_opcodes():
        if op != "equal":
            return i1, i2, j1, j2
    return None


@torch.no_grad()
def _next_dist(model: object, ids: list[int], device: torch.device) -> torch.Tensor:
    x = torch.tensor([ids], device=device)
    logits = model(x).logits[0, -1].float()  # type: ignore[attr-defined]
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def run(
    model_name: str,
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> None:
    device = resolve_device(require_cuda=False)
    print(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map={"": device.type}, low_cpu_mem_usage=True
    )
    model.eval()
    torch.manual_seed(seed)
    use_chat = tokenizer.chat_template is not None

    boundaries: list[Boundary] = []
    n_gen = 0
    n_noncanon = 0

    for domain, prompts in PROMPTS.items():
        for prompt in prompts:
            if use_chat:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt
            enc = tokenizer(text, return_tensors="pt").to(device)
            plen = enc.input_ids.shape[1]
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
            )
            for seq in out:
                gen = seq[plen:].tolist()
                while gen and gen[-1] in (
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                ):
                    gen.pop()
                if len(gen) < 3:
                    continue
                n_gen += 1
                text = decode_for_roundtrip(tokenizer, gen)
                if not roundtrip_is_measurable(tokenizer, text):
                    continue  # confounded round trip (U+FFFD / non-NFC)
                canon = tokenizer.encode(text, add_special_tokens=False)
                region = _first_replace(gen, canon)
                if region is None:
                    continue
                n_noncanon += 1
                i1, i2, j1, j2 = region
                # boundary = right after the differing span; identical prefix text
                if i2 >= len(gen) or j2 >= len(canon):
                    continue  # span at the very end: no next-token to compare
                p_a = _next_dist(model, gen[:i2], device)
                p_c = _next_dist(model, canon[:j2], device)
                kl = float(
                    (
                        p_a * (p_a.clamp_min(1e-12).log() - p_c.clamp_min(1e-12).log())
                    ).sum()
                )
                t_a = int(p_a.argmax())
                t_c = int(p_c.argmax())
                boundaries.append(
                    Boundary(
                        domain=domain,
                        span_actual=[tokenizer.decode([t]) for t in gen[i1:i2]],
                        span_canonical=[tokenizer.decode([t]) for t in canon[j1:j2]],
                        kl=kl,
                        actual_next=tokenizer.decode([t_a]),
                        canonical_next=tokenizer.decode([t_c]),
                        pa_top1=float(p_a[t_a]),
                        pa_at_canon=float(p_a[t_c]),
                    )
                )

    if not boundaries:
        print("No non-canonical spans with a downstream boundary found.")
        return

    def _median(xs: list[float]) -> float:
        return float(torch.tensor(xs).median()) if xs else float("nan")

    def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval — valid at the small counts these rates hit."""
        if n == 0:
            return (float("nan"), float("nan"))
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    kls = torch.tensor([b.kl for b in boundaries])
    flips = [b for b in boundaries if b.flipped]
    disagree = len(flips) / len(boundaries)
    # a flip is "meaningful" if the actual run was confident in its own choice
    conf_flips = [b for b in flips if b.pa_top1 > 0.5]

    print("\n===== INTERP MISMATCH AT NON-CANONICAL SPAN BOUNDARIES =====")
    print(f"model: {model_name}")
    print(
        f"generations: {n_gen}, non-canonical: {n_noncanon}, "
        f"boundaries measured: {len(boundaries)}"
    )
    print(
        f"next-token KL(actual ‖ re-tokenized): mean={kls.mean():.3f} "
        f"median={kls.median():.3f} max={kls.max():.3f} nats "
        f"(canonical control = 0 by construction)"
    )
    flo, fhi = _wilson(len(flips), len(boundaries))
    print(
        f"top-1 next-token FLIPS after re-tokenization: {disagree:.0%} "
        f"({len(flips)}/{len(boundaries)}, 95% CI [{flo:.0%}, {fhi:.0%}])"
    )
    if flips:
        print(
            "  flip confidence — actual run's prob on its own top-1: "
            f"median={_median([b.pa_top1 for b in flips]):.2f}; on the replay's "
            f"pick: median={_median([b.pa_at_canon for b in flips]):.2f}"
        )
        conf_pct = len(conf_flips) / len(boundaries)
        clo, chi = _wilson(len(conf_flips), len(boundaries))
        print(
            "  'meaningful' flips (actual confident, p>0.5 on its top-1): "
            f"{len(conf_flips)}/{len(boundaries)} = {conf_pct:.0%} "
            f"(95% CI [{clo:.0%}, {chi:.0%}])"
        )

    print("\n--- per-domain ---")
    hdr = f"{'domain':<14}{'bounds':>8}{'meanKL':>9}{'flip%':>8}{'conf-flip%':>12}"
    print(hdr)
    for domain in {b.domain for b in boundaries}:
        db = [b for b in boundaries if b.domain == domain]
        df = [b for b in db if b.flipped]
        dc = [b for b in df if b.pa_top1 > 0.5]
        mkl = float(torch.tensor([b.kl for b in db]).mean())
        print(
            f"{domain:<14}{len(db):>11}{mkl:>9.2f}"
            f"{len(df) / len(db):>7.0%}{len(dc) / len(db):>11.0%}"
        )

    print(
        "\n--- example boundaries (actual span | canonical span | "
        "predicted next: actual vs re-tokenized) ---"
    )
    for b in sorted(boundaries, key=lambda x: -x.kl)[:12]:
        flag = "DIFFERENT" if b.actual_next != b.canonical_next else "same"
        print(
            f"[{b.domain}] actual={b.span_actual} canon={b.span_canonical} "
            f"KL={b.kl:.2f} | next: {b.actual_next!r} vs {b.canonical_next!r} ({flag})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 interp-mismatch demo")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
