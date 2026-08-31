"""Arm C task: word reversal — the compute-aligned task, plus its metrics.

Reversal is serial and its output order tracks computation order (each next
character needs one more step of a right-to-left scan), which is the property
the addition null identified as required for the extra-decode-steps channel to
pay. The reward reads the DECODED completion, so segmentation stays a free
variable the reward cannot see. See EXPERIMENT_PLAN.md, Arm C.

Three attractors over the emitted letter run, mirroring the expansion arm:

- **all-single-char** — one letter per token, the compute-buying form
- **greedy-longest** — reward-dense chunks, what a memorising policy prefers
- **canonical** — BPE's merge-order segmentation of the reversed surface

Correctness/effort metrics are first-class here: the arm is only evidence if
the policy is actually attempting (and improving at) the task.
"""

from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from retok_rl.metrics import leading_correct, reference_matches

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Common English words, lengths 3-8. Length is the difficulty knob (the
# analogue of the expansion arm's divisor class): the measured Qwen2.5-3B
# baseline near-masters length 3 and degrades monotonically to length 8.
WORDS_BY_LEN: dict[int, list[str]] = {
    3: [
        "cat",
        "dog",
        "sun",
        "map",
        "box",
        "red",
        "car",
        "pen",
        "cup",
        "hat",
        "bed",
        "sky",
        "ice",
        "arm",
        "leg",
        "egg",
        "jam",
        "key",
        "fox",
        "owl",
        "bee",
        "ant",
        "cow",
        "pig",
        "rat",
        "bag",
        "bat",
        "can",
        "cap",
        "dot",
        "ear",
        "eye",
        "fan",
        "fig",
        "gem",
        "hen",
        "hut",
        "ink",
        "jar",
        "kit",
        "lip",
        "log",
        "man",
        "mud",
        "net",
        "nut",
        "oak",
        "pan",
    ],
    4: [
        "fish",
        "star",
        "book",
        "lamp",
        "tree",
        "milk",
        "door",
        "wind",
        "rain",
        "snow",
        "fire",
        "gold",
        "ship",
        "bird",
        "cake",
        "corn",
        "desk",
        "drum",
        "farm",
        "frog",
        "game",
        "hand",
        "iron",
        "king",
        "lake",
        "leaf",
        "lion",
        "moon",
        "nest",
        "pond",
        "rock",
        "roof",
        "room",
        "root",
        "rose",
        "sand",
        "seed",
        "song",
        "soup",
        "wall",
        "wave",
        "wolf",
        "wood",
        "wool",
        "yard",
        "bell",
        "belt",
        "bone",
    ],
    5: [
        "house",
        "plant",
        "river",
        "cloud",
        "bread",
        "stone",
        "apple",
        "beach",
        "brain",
        "chair",
        "dance",
        "eagle",
        "earth",
        "field",
        "fruit",
        "glass",
        "grape",
        "green",
        "heart",
        "horse",
        "juice",
        "knife",
        "lemon",
        "light",
        "money",
        "mouse",
        "music",
        "night",
        "ocean",
        "paint",
        "paper",
        "peach",
        "piano",
        "pilot",
        "queen",
        "radio",
        "shark",
        "sheep",
        "shelf",
        "smile",
        "snake",
        "spoon",
        "sugar",
        "table",
        "tiger",
        "train",
        "water",
        "wheel",
    ],
    6: [
        "garden",
        "planet",
        "bottle",
        "memory",
        "silver",
        "window",
        "animal",
        "basket",
        "bridge",
        "butter",
        "camera",
        "candle",
        "carpet",
        "castle",
        "cheese",
        "circle",
        "coffee",
        "cotton",
        "dinner",
        "doctor",
        "dragon",
        "flower",
        "forest",
        "guitar",
        "hammer",
        "island",
        "jacket",
        "jungle",
        "kitten",
        "ladder",
        "lizard",
        "marble",
        "market",
        "mirror",
        "monkey",
        "mother",
        "museum",
        "orange",
        "pencil",
        "pepper",
        "pillow",
        "pocket",
        "rabbit",
        "rocket",
        "school",
        "season",
        "sister",
        "spider",
        "spring",
        "stream",
        "street",
        "summer",
        "temple",
        "ticket",
        "tunnel",
        "turtle",
        "valley",
        "violin",
        "winter",
        "yellow",
    ],
    7: [
        "kitchen",
        "library",
        "machine",
        "morning",
        "picture",
        "rainbow",
        "bicycle",
        "blanket",
        "brother",
        "cabinet",
        "ceiling",
        "channel",
        "chicken",
        "chimney",
        "country",
        "crystal",
        "curtain",
        "diamond",
        "dolphin",
        "drawing",
        "evening",
        "factory",
        "feather",
        "fiction",
        "giraffe",
        "grammar",
        "harvest",
        "history",
        "holiday",
        "journey",
        "lantern",
        "leather",
        "lettuce",
        "mineral",
        "monster",
        "octopus",
        "orchard",
        "package",
        "pattern",
        "penguin",
        "pyramid",
        "rooster",
        "soldier",
        "station",
        "teacher",
        "theater",
        "thunder",
        "traffic",
        "village",
        "vitamin",
        "weather",
        "whisper",
    ],
    8: [
        "keyboard",
        "notebook",
        "mountain",
        "sandwich",
        "elephant",
        "computer",
        "aircraft",
        "airplane",
        "alphabet",
        "birthday",
        "building",
        "calendar",
        "campfire",
        "champion",
        "daughter",
        "dinosaur",
        "distance",
        "engineer",
        "envelope",
        "festival",
        "football",
        "fountain",
        "hospital",
        "lavender",
        "magazine",
        "medicine",
        "mushroom",
        "painting",
        "paradise",
        "passport",
        "pavement",
        "platform",
        "princess",
        "question",
        "rainfall",
        "sailboat",
        "scissors",
        "shoulder",
        "sidewalk",
        "skeleton",
        "squirrel",
        "stairway",
        "sunshine",
        "thursday",
        "tomorrow",
        "treasure",
        "umbrella",
        "universe",
        "vacation",
        "wildfire",
    ],
}


def split_words(n_held_out: int, seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministic train/held-out word split, stratified by length.

    Stratified for the same reason as the divisor split: an all-length-8
    held-out set would make the generalisation eval harder than the training
    distribution.
    """
    import random

    rng = random.Random(seed)
    total = sum(len(ws) for ws in WORDS_BY_LEN.values())
    held: list[str] = []
    for length in sorted(WORDS_BY_LEN):
        pool = sorted(WORDS_BY_LEN[length])
        rng.shuffle(pool)
        k = max(1, round(n_held_out * len(pool) / total))
        held.extend(pool[:k])
    held_set = set(held)
    train = tuple(
        w
        for length in sorted(WORDS_BY_LEN)
        for w in WORDS_BY_LEN[length]
        if w not in held_set
    )
    return train, tuple(sorted(held_set))


@dataclass(frozen=True)
class Example:
    word: str
    prompt: str
    target: str


PROMPT_TEMPLATE = (
    'Spell the word "{word}" backwards. Reply with only the reversed word, '
    "in lowercase, with no spaces or punctuation."
)


def build_reversal(
    words: tuple[str, ...], tok: PreTrainedTokenizerBase
) -> list[Example]:
    """One example per word; the prompt is the model's own chat template.

    Templating happens here (not in the trainer) so the eval callback and the
    trainer see the identical prompt string, and so the token IDs that reach
    generate() are unambiguous.
    """
    out = []
    for w in words:
        if getattr(tok, "chat_template", None):
            msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(word=w)}]
            prompt = str(
                tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )
        else:
            # Base models (gpt2 smoke tests): plain prompt, and no trailing
            # space — the documented footgun that invalidated earlier runs.
            prompt = PROMPT_TEMPLATE.format(word=w) + "\nAnswer:"
        out.append(Example(word=w, prompt=prompt, target=w[::-1]))
    return out


def letter_prefix(text: str) -> str:
    """Leading alphabetic run of a completion, ignoring leading whitespace."""
    out = ""
    for ch in text.lstrip():
        if ch.isalpha():
            out += ch.lower()
        else:
            break
    return out


def tokens_covering_letters(
    tok: PreTrainedTokenizerBase, token_ids: list[int]
) -> tuple[list[int], str]:
    """The prefix of ``token_ids`` whose surface is the leading letter run.

    The surface is returned case-PRESERVED: segmentation metrics must compare
    the emitted tokens against the canonical encoding of the string that was
    actually produced. (Run 5 originally lowercased here, which misfiled
    canonically-tokenized uppercase output as "other".) Compliance checks
    lowercase at the comparison site instead.
    """
    kept: list[int] = []
    surface = ""
    for t in token_ids:
        piece = tok.decode([t])
        probe = (surface + piece).lstrip()
        if probe and all(c.isalpha() for c in probe):
            kept.append(t)
            surface += piece
        else:
            break
    return kept, surface.lstrip()


def classify_letters(
    tok: PreTrainedTokenizerBase, token_ids: list[int], letters: str
) -> str:
    """Which attractor does this token sequence match, over the letter run?

    ``letters`` must be the case-preserved surface of ``token_ids`` — the
    comparison is against the canonical encoding of the string actually
    produced. Only meaningful for compliant rollouts (produced == target):
    wrong text is just wrong, and contributes nothing to segmentation
    metrics; canonicality of arbitrary emitted text is measured separately
    by the round-trip check in ``summarise_reversal``.
    """
    m = reference_matches(tok, token_ids, letters)
    if m["all-single-digit"]:
        return "all-single-char"
    for name in ("canonical", "greedy-longest"):
        if m[name]:
            return name
    return "other"


def letter_overlap(produced: str, target: str) -> float:
    """Multiset letter overlap with the target, as a fraction of the target.

    Distinguishes "reversing badly" (right letters, wrong order → high
    overlap) from "ignoring the task" (unrelated text → low overlap).
    """
    if not target:
        return 0.0
    inter = Counter(produced) & Counter(target)
    return sum(inter.values()) / len(target)


Rollout = tuple[list[int], str]  # (full completion token ids, target)


def summarise_reversal(
    tok: PreTrainedTokenizerBase, rollouts: list[Rollout]
) -> dict[str, float]:
    """Aggregate correctness, effort, and tokenization stats over rollouts.

    Tokenization is measured two independent ways, and the distinction is
    load-bearing (an unexpected-but-canonical token is not a non-canonical
    token):

    - **Round-trip canonicality** of whatever the model emitted:
      ``encode(decode(ids)) != ids`` — the project's core metric, applied to
      the full completion (specials stripped) and, separately, to the letter
      run. Reported per-generation AND per-token, because the per-generation
      flag grows mechanically with completion length (non-recovering BPE).
    - **Attractor mix** over COMPLIANT rollouts only (produced == target,
      case-insensitively): which segmentation of the correct answer was
      used. Wrong text contributes zero here; its only effect is on the
      correctness metrics and reward.

    The primary segmentation metrics are the ``correct_region/*`` family:
    per-token, computed only over tokens that lie entirely within the
    correct leading prefix of the answer (the characters that match the
    target). Wrong characters and everything after them are excluded by
    construction, so junk cannot move these numbers in either direction.
    ``pos_single_char/{i}`` gives the per-position view: among rollouts
    whose i-th answer character is correct, how often is that character
    carried by a length-1 token.
    """
    special = set(tok.all_special_ids)
    attractors: Counter[str] = Counter()
    matches: Counter[str] = Counter()
    single_toks = total_toks = 0
    reward_sum = overlap_sum = len_ratio_sum = 0.0
    exact = attempted = empty = 0
    gen_nc = tok_bad = tok_total = excluded = 0
    ans_nc = ans_n = ans_tok_bad = ans_tok_total = 0
    cr_single = cr_total = cr_nc = cr_n = 0
    pos_single: Counter[int] = Counter()
    pos_n: Counter[int] = Counter()
    for ids, target in rollouts:
        core = [t for t in ids if t not in special]
        kept, surface = tokens_covering_letters(tok, core)
        produced = surface.lower()
        reward_sum += leading_correct(produced, target)
        overlap_sum += letter_overlap(produced, target)
        len_ratio_sum += len(produced) / max(1, len(target))
        compliant = produced == target
        exact += compliant
        attempted += len(produced) >= 3
        if not kept:
            empty += 1
        # round-trip canonicality of the emitted stream, junk and all
        if core:
            text = tok.decode(core)
            if "�" in text:
                excluded += 1
            else:
                canon = tok.encode(text, add_special_tokens=False)
                tok_total += len(core)
                if canon != core:
                    gen_nc += 1
                    for op, i1, i2, _, _ in difflib.SequenceMatcher(
                        a=core, b=canon, autojunk=False
                    ).get_opcodes():
                        if op != "equal":
                            tok_bad += i2 - i1
        # round-trip canonicality of the letter run alone (case-preserved)
        if kept and "�" not in surface:
            ans_n += 1
            canon_run = tok.encode(surface, add_special_tokens=False)
            canon_run_sp = tok.encode(" " + surface, add_special_tokens=False)
            ans_nc += kept != canon_run and kept != canon_run_sp
            # per-token, region-only: round-trip the letter run's own raw
            # surface (leading whitespace included), so text after the
            # answer can neither dilute nor inflate the rate
            raw_run = tok.decode(kept)
            canon_raw = tok.encode(raw_run, add_special_tokens=False)
            ans_tok_total += len(kept)
            if canon_raw != kept:
                for op, i1, i2, _, _ in difflib.SequenceMatcher(
                    a=kept, b=canon_raw, autojunk=False
                ).get_opcodes():
                    if op != "equal":
                        ans_tok_bad += i2 - i1
        # segmentation-of-the-answer: compliant rollouts only
        if compliant and kept:
            attractors[classify_letters(tok, kept, surface)] += 1
            for name, hit in reference_matches(tok, kept, surface).items():
                matches[name] += hit
        for i, t in enumerate(kept):
            piece = tok.decode([t])
            piece = piece.strip() if i == 0 else piece
            total_toks += 1
            single_toks += len(piece) == 1
        # correct-region metrics: tokens lying entirely within the correct
        # leading prefix (characters matching the target, case-insensitive)
        n_correct = leading_correct(produced, target)
        if n_correct and kept:
            cr_ids: list[int] = []
            cr_surface = ""
            consumed = 0
            for i, t in enumerate(kept):
                piece = tok.decode([t])
                stripped = piece.lstrip() if i == 0 else piece
                if consumed + len(stripped) > n_correct:
                    break  # token straddles the correct/incorrect boundary
                # per-position: character j (1-indexed) carried by this token
                for j in range(consumed + 1, consumed + len(stripped) + 1):
                    pos_n[j] += 1
                    pos_single[j] += len(stripped) == 1
                cr_ids.append(t)
                cr_surface += stripped
                cr_total += 1
                cr_single += len(stripped) == 1
                consumed += len(stripped)
            if cr_ids and "�" not in cr_surface:
                cr_n += 1
                canon_cr = tok.encode(cr_surface, add_special_tokens=False)
                canon_cr_sp = tok.encode(" " + cr_surface, add_special_tokens=False)
                cr_nc += cr_ids != canon_cr and cr_ids != canon_cr_sp
    n = max(1, len(rollouts))
    n_meas = max(1, len(rollouts) - excluded)
    n_comp = max(1, sum(attractors.values()))
    out: dict[str, float] = {
        "mean_reward": reward_sum / n,
        "exact_match": exact / n,
        "attempted": attempted / n,
        "mean_len_ratio": len_ratio_sum / n,
        "letter_overlap": overlap_sum / n,
        "empty": empty / n,
        "frac_single_char_tokens": single_toks / max(1, total_toks),
        "roundtrip/gen_non_canonical": gen_nc / n_meas,
        "roundtrip/tok_non_canonical": tok_bad / max(1, tok_total),
        "roundtrip/answer_non_canonical": ans_nc / max(1, ans_n),
        "roundtrip/answer_tok_non_canonical": ans_tok_bad / max(1, ans_tok_total),
        "mean_answer_tokens": ans_tok_total / max(1, ans_n),
        "roundtrip/excluded": excluded / n,
        "n_compliant": float(sum(attractors.values())),
        "correct_region/frac_single_char": cr_single / max(1, cr_total),
        "correct_region/non_canonical": cr_nc / max(1, cr_n),
        "correct_region/mean_tokens": cr_total / max(1, cr_n),
    }
    for name in ("all-single-char", "canonical", "greedy-longest", "other"):
        out[f"attractor_compliant/{name}"] = attractors[name] / n_comp
    # non-exclusive: a compliant answer can match several references
    # (canonical == greedy-longest for 87% of compliant surfaces in Run 5),
    # so these do not sum to 100%. Denominator: compliant rollouts.
    for src, name in (
        ("all-single-digit", "all-single-char"),
        ("canonical", "canonical"),
        ("greedy-longest", "greedy-longest"),
    ):
        out[f"match_compliant/{name}"] = matches[src] / n_comp
    for j in range(1, 5):
        out[f"pos_single_char/{j}"] = pos_single[j] / max(1, pos_n[j])
        out[f"pos_n/{j}"] = float(pos_n[j])
    return out
