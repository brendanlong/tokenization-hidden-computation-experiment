"""Figures for the retok experiment.

- ``mechanism``: the hero schematic — the model emits the answer over 3 decode
  positions; storing it as text and re-encoding collapses it to 1, so
  per-position analysis runs on positions that never existed.
- ``calibration``: the text-only detector — the re-tokenized transcript is
  always correct, yet the model assigns its one-token form ~0.003.
- ``width_sweep``: the appendix sweep locating the dim=16 regime.
- ``phase2_rates``: wild-caught non-canonical rate by model and domain.

Regenerate: ``uv run python -m retok.figures``. Numbers are the
official runs recorded in RESULTS.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

INK = "#222222"
MUTED = "#666666"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
SURFACE = "#fcfcfb"

# Official width sweep (base-10, D=3, L=2, seed 0, 15M examples). See RESULTS.md.
DIMS = [8, 12, 16, 24, 32, 64]
SERIES = {
    "CoT (3 tokens)": ([0.9, 89.8, 100.0, 100.0, 100.0, 100.0], "#0072B2"),
    "one-step, per-digit": ([1.0, 9.9, 10.5, 100.0, 59.5, 100.0], "#009E73"),
    "one-step, merged token": ([0.3, 0.9, 1.5, 2.0, 3.8, 97.5], "#E69F00"),
}
CHOSEN_DIM = 16


def plot_width_sweep(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("white")

    for label, (ys, color) in SERIES.items():
        ax.plot(
            DIMS,
            ys,
            marker="o",
            markersize=7,
            linewidth=2.0,
            color=color,
            label=label,
            zorder=3,
        )

    # highlight the chosen regime
    ax.axvline(CHOSEN_DIM, color="#888", linewidth=1.0, linestyle=":", zorder=1)
    ax.annotate(
        "chosen model:\ndim=16, only CoT works",
        xy=(CHOSEN_DIM, 62),
        xytext=(CHOSEN_DIM * 1.05, 62),
        fontsize=8.5,
        color="#444",
        ha="left",
        va="center",
    )

    # legend in the empty lower-center band (orange stays low until dim~32,
    # green/blue are up at 100) — identity is never color-alone.
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.52, 0.2),
        framealpha=0.92,
        edgecolor="#dddddd",
        fontsize=9,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlim(7, 72)
    ax.set_ylim(-4, 108)
    ax.set_xlabel("model width (embedding dim), 2 layers")
    ax.set_ylabel("answer accuracy (%)")
    ax.set_title(
        "Digit-by-digit derivation succeeds where one pass fails\n"
        "3-digit reversed addition · width sweep",
        fontsize=11,
        loc="left",
    )
    ax.grid(True, which="major", axis="both", color="#e8e8e6", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


# Phase-2 non-canonical generation rate (% of generations with >=1 non-canonical
# span), by domain and model. See RESULTS.md "Phase-2".
# CONTROLLED measurement: pure sampling (top_k=0, top_p=1.0, rep_penalty=1.0),
# clean_up_tokenization_spaces=False, U+FFFD and non-NFC excluded. temp 1.0.
# See RESULTS.md — earlier numbers were confounded by inherited sampling params
# and Llama-only decode post-processing.
#
# PER-TOKEN is the primary metric. BPE is non-recovering (Chatzi et al.): once a
# sequence goes non-canonical every extension stays non-canonical, so the
# per-generation rate is monotone in length and SATURATES at 100% — it is not
# comparable across generation lengths. Per-token has neither problem.
PHASE2_DOMAINS = [
    "english",
    "arithmetic",
    "code",
    "multiling.\n(Latin)",
    "multiling.\n(Cyrillic)",
    "multiling.\n(CJK)",
]
PHASE2_RATES = {  # model -> per-domain % of TOKENS non-canonical ; color
    "GPT-2 (124M, fp32)": ([0.89, 1.41, 1.54, 3.28, 0.43, 1.09], "#E69F00"),
    "Llama-3.2-1B": ([0.03, 0.12, 0.35, 1.28, 4.06, 5.20], "#0072B2"),
    "Qwen2.5-1.5B": ([0.00, 0.05, 0.08, 0.76, 0.25, 0.59], "#009E73"),
    "Gemma-2-2b (fp32)": ([0.07, 0.06, 0.12, 0.45, 0.66, 0.54], "#CC79A7"),
    "Llama-3.2-3B": ([0.00, 0.05, 0.05, 0.99, 1.98, 2.93], "#D55E00"),
    "Llama-3.1-8B": ([0.03, 0.03, 0.03, 0.42, 1.31, 1.21], "#56B4E9"),
    "gpt-oss-20b": ([0.03, 0.00, 0.00, 0.11, 0.14, 0.00], "#333333"),
}

# Scale ladder: Llama family, identical tokenizer. Rate falls with size.
LADDER_LABELS = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"]
LADDER_X = [1, 3, 8]
LADDER = {
    "multiling. CJK": ([5.20, 2.93, 1.21], "#0072B2"),
    "multiling. Cyrillic": ([4.06, 1.98, 1.31], "#56B4E9"),
    "multiling. Latin": ([1.28, 0.99, 0.42], "#009E73"),
    "code": ([0.35, 0.05, 0.03], "#E69F00"),
}


def plot_scale_ladder(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor("white")
    # CJK and Cyrillic converge at 8B, so label at the LEFT end where the
    # series are well separated, and keep a legend for redundancy.
    for label, (ys, color) in LADDER.items():
        ax.plot(
            LADDER_X,
            ys,
            marker="o",
            markersize=7,
            linewidth=2.0,
            color=color,
            label=label,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(LADDER_X[0], ys[0]),
            xytext=(-8, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            fontsize=8.5,
            color=color,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(LADDER_X)
    ax.set_xticklabels(["1B", "3B", "8B"])
    ax.set_xlim(0.42, 10)
    ax.set_ylim(0, 5.8)
    ax.set_xlabel("model size (Llama family, identical tokenizer)")
    ax.set_ylabel("emitted tokens that are non-canonical (%)")
    ax.set_title(
        "Non-canonical generation falls with scale\n"
        "Same tokenizer throughout, so this is a model effect",
        fontsize=10.5,
        loc="left",
    )
    ax.grid(True, color="#e8e8e6", linewidth=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#cccccc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


# Temperature sweep (code+multilingual prompts, 48 gens/cell; the temp-0 cell is
# 8 deterministic gens, since greedy repeats). See RESULTS.md.
#
# NOTE the temp-0 zeros below are subset-limited, not a true zero: re-running
# greedy over the full 25-prompt set finds 0.08% pooled across four models
# (RESULTS.md, "Greedy decoding is NOT 0%"). The 8-prompt sweep set contains no
# arithmetic prompts, which is where GPT-2's greedy hits are. Kept as measured
# so this curve stays one internally-consistent experiment; the annotation on
# the plot carries the correction.
PHASE2_TEMPS = [0.0, 0.7, 1.0, 1.5, 2.0]
# CONTROLLED (pure sampling). Per-generation here is fine: it's a within-model
# comparison at FIXED generation length, so the length-saturation problem that
# rules per-generation out for cross-domain comparison doesn't apply.
PHASE2_TEMP_RATES = {  # per-TOKEN % (we report per-token everywhere; the
    # per-generation rate saturates with length and isn't comparable)
    "Llama-3.2-1B": ([0.00, 0.22, 0.95, 3.68, 4.30], "#0072B2"),
    "Qwen2.5-1.5B": ([0.00, 0.22, 0.42, 3.33, 3.77], "#009E73"),
}


def plot_temperature(out_path: Path) -> None:
    """Non-canonical rate is largely a tail-sampling phenomenon."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor("white")
    for model, (rates, color) in PHASE2_TEMP_RATES.items():
        ax.plot(
            PHASE2_TEMPS,
            rates,
            marker="o",
            markersize=7,
            linewidth=2.0,
            color=color,
            label=model,
            zorder=3,
        )
    ax.set_xticks(PHASE2_TEMPS)
    ax.set_xlabel("sampling temperature")
    ax.set_ylabel("emitted tokens that are non-canonical (%)")
    ax.set_ylim(-0.2, 5.0)
    ax.set_title(
        "Non-canonical generation is largely a tail-sampling effect\n"
        "~0.1% under greedy decoding; 3-4% of tokens by temperature 1.5",
        fontsize=10.5,
        loc="left",
    )
    # The 0.0 points read as an exact zero, which a wider prompt set refutes.
    ax.annotate(
        "greedy is 0% on these 8 prompts, but\n"
        "0.08% over the full 25-prompt set\n"
        "(6/99 gens) — low, not immune",
        xy=(0.0, 0.0),
        xytext=(0.28, 1.45),
        fontsize=8.2,
        color=MUTED,
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.9},
    )
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="#dddddd", fontsize=9)
    ax.grid(True, color="#e8e8e6", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def plot_phase2_rates(out_path: Path) -> None:
    n_models = len(PHASE2_RATES)
    x = list(range(len(PHASE2_DOMAINS)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("white")
    for m, (model, (rates, color)) in enumerate(PHASE2_RATES.items()):
        offs = (m - (n_models - 1) / 2) * width
        bars = ax.bar(
            [xi + offs for xi in x],
            rates,
            width * 0.92,
            color=color,
            label=model,
            zorder=3,
        )
        for bar, r in zip(bars, rates, strict=True):
            ax.annotate(
                f"{r:g}",
                xy=(bar.get_x() + bar.get_width() / 2, r),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE2_DOMAINS, fontsize=8.5)
    ax.set_ylabel("emitted tokens that are non-canonical (%)")
    ax.set_ylim(0, 6.2)
    ax.set_title(
        "Real models generate non-canonically, unprompted\n"
        "Per-token rate, as-released precision, pure sampling at temp 1.0.\n"
        "Language matters more than model; no model tested is immune",
        fontsize=10.5,
        loc="left",
    )
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="#dddddd", fontsize=9)
    ax.grid(True, axis="y", color="#e8e8e6", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


# Causal-contamination decay (phase2_decay, Llama-3.2-1B, 49 spans).
DECAY_D = [0, 1, 2, 4, 8, 16, 32, 64]
DECAY_MEDKL = [0.227, 0.062, 0.025, 0.016, 0.007, 0.004, 0.003, 0.002]
DECAY_FLIP = [50, 27, 14, 20, 12, 9, 11, 15]


def plot_decay(out_path: Path) -> None:
    """Magnitude dilutes fast; the argmax-flip rate plateaus rather than vanishing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("symlog", base=2, linthresh=1)
        ax.set_xticks(DECAY_D)
        ax.set_xticklabels([str(d) for d in DECAY_D], fontsize=8.5)
        ax.set_xlabel("tokens past the non-canonical span")
        ax.grid(True, color="#e8e8e6", linewidth=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#cccccc")
    fig.patch.set_facecolor("white")

    ax1.plot(
        DECAY_D,
        DECAY_MEDKL,
        marker="o",
        markersize=7,
        linewidth=2.0,
        color=BLUE,
        zorder=3,
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("median next-token KL (nats)")
    ax1.set_title("Magnitude dilutes ~100x by 16 tokens", fontsize=10, loc="left")

    ax2.plot(
        DECAY_D,
        DECAY_FLIP,
        marker="o",
        markersize=7,
        linewidth=2.0,
        color=ORANGE,
        zorder=3,
    )
    ax2.set_ylim(0, 56)
    ax2.set_ylabel("top-1 next-token flips (%)")
    ax2.set_title(
        "...but the flip rate plateaus near 10%, it doesn't vanish",
        fontsize=10,
        loc="left",
    )
    ax2.axhspan(9, 15, color=ORANGE, alpha=0.10, zorder=1)

    fig.suptitle(
        "A single non-canonical token perturbs every later position",
        fontsize=11.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def _token_box(
    ax: object,
    x: float,
    y: float,
    w: float,
    label: str,
    color: str,
    *,
    fontsize: float = 11,
) -> None:
    """Draw one token as a rounded box with its surface text."""
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        0.42,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color + "22",
        zorder=3,
    )
    ax.add_patch(box)  # type: ignore[attr-defined]
    ax.text(  # type: ignore[attr-defined]
        x + w / 2,
        y + 0.21,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        zorder=4,
        family="monospace",
    )


def plot_mechanism(out_path: Path) -> None:
    """Hero schematic: 3 emitted decode positions collapse to 1 on re-encoding."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        0.1, 4.85, "What the model ran", fontsize=11.5, color=INK, fontweight="bold"
    )
    ax.text(
        0.1,
        4.55,
        "3 decode steps — one forward pass per digit",
        fontsize=9.5,
        color=MUTED,
    )
    prompt = ["<bos>", "1", "5", "+", "6", "4", "="]
    x = 0.1
    for t in prompt:
        w = 0.52 + 0.12 * max(0, len(t) - 1)
        _token_box(ax, x, 3.9, w, t, MUTED, fontsize=9.5)
        x += w + 0.09
    for t in ["9", "7", "0"]:
        _token_box(ax, x, 3.9, 0.52, t, BLUE)
        x += 0.61
    ax.annotate(
        "3 positions to probe\n(carry computed step by step)",
        xy=(x - 0.3, 3.86),
        xytext=(x + 0.35, 3.45),
        ha="left",
        va="center",
        fontsize=9,
        color=BLUE,
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2},
    )

    # the lossy round trip
    ax.annotate(
        "",
        xy=(0.9, 2.25),
        xytext=(0.9, 3.05),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.8},
    )
    ax.text(
        1.15,
        2.62,
        'store as text  "…= 970"  →  re-encode (BPE longest match)',
        fontsize=9.5,
        color=INK,
        va="center",
    )

    ax.text(
        0.1,
        2.05,
        "What the stored transcript says",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.1,
        1.75,
        "1 decode step — the derivation is gone",
        fontsize=9.5,
        color=MUTED,
    )
    x = 0.1
    for t in prompt:
        w = 0.52 + 0.12 * max(0, len(t) - 1)
        _token_box(ax, x, 1.1, w, t, MUTED, fontsize=9.5)
        x += w + 0.09
    _token_box(ax, x, 1.1, 1.0, "970", ORANGE)
    ax.annotate(
        "1 position — same string, different tokens",
        xy=(x + 1.02, 1.31),
        xytext=(x + 1.3, 1.31),
        ha="left",
        va="center",
        fontsize=9,
        color=ORANGE,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.2},
    )

    ax.text(
        0.1,
        0.42,
        "per-position analysis (logit lens, probes, serial-depth accounting) "
        "now runs on positions that never existed",
        fontsize=9.5,
        color=INK,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": SURFACE,
            "edgecolor": "#dddddd",
        },
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def plot_calibration(out_path: Path) -> None:
    """The text-only detector: transcripts are always right, one-step prob ~0."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor("white")
    labels = [
        "re-tokenized transcripts\nthat are correct",
        "model's probability of\nemitting that answer in\none token",
    ]
    values = [100.0, 0.3]
    colors = [BLUE, ORANGE]
    bars = ax.bar(labels, values, width=0.5, color=colors, zorder=3)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:g}%",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
    ax.set_ylim(0, 118)
    ax.set_ylabel("percent")
    ax.set_title(
        "The calibration detector (text + logprobs only)\n"
        "Transcripts are 100% correct at ~0.003 per-transcript probability —\n"
        "astronomically unlikely if they were really one-step generations",
        fontsize=10.5,
        loc="left",
    )
    ax.grid(True, axis="y", color="#e8e8e6", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main() -> None:
    figdir = Path(__file__).resolve().parent.parent / "figures"
    plot_mechanism(figdir / "mechanism.png")
    plot_calibration(figdir / "calibration.png")
    plot_width_sweep(figdir / "width_sweep.png")
    plot_phase2_rates(figdir / "phase2_rates.png")
    plot_temperature(figdir / "phase2_temperature.png")
    plot_decay(figdir / "phase2_decay.png")
    plot_scale_ladder(figdir / "phase2_scale.png")


if __name__ == "__main__":
    main()
