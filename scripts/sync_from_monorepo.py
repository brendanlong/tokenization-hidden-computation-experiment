"""Sync this release from the private monorepo it was extracted from.

This repo is not a copy of `experiments/retok/` — it is a *transform* of it:
imports are flattened, and a handful of files carry release-only changes
(HuggingFace checkpoint resolution instead of S3, no run-name collision guards,
extra `--all-published` entry points). Every hand-run sync so far has reverted
part of that transform, silently:

- a path-rewriting `sed` renamed the S3 bucket *inside* RESULTS.md, a file whose
  whole point is being verbatim;
- a wholesale file copy restored `upload_checkpoint_to_s3` in `training.py`,
  dropped `--all-published` from `phase2_verify.py`, and reintroduced `CLAUDE.md`
  references and an `s3://` docstring.

So the transform lives here instead of in someone's head. Run:

    uv run python scripts/sync_from_monorepo.py --monorepo ../experiments

It is deliberately loud: if a patch below no longer matches, that means upstream
changed the code the patch targets, and the sync **stops** rather than quietly
producing a release that is missing a strip.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXPERIMENT = "retok"

# Flattening: `experiments.retok.X` -> `retok.X`, `shared.X` -> `common.X`.
# Only ever applied to *module paths*, never to prose or URIs — an earlier sed
# on `experiments/retok/` also matched inside `s3://brendanlong-experiments/retok/`.
SHARED_MODULES = ("checkpoint", "config", "gpu", "schedule", "streaming", "wandb_utils")

# Files that carry release-only divergence. Never copied: the release version is
# authoritative. If upstream changes one of these, a human has to port it.
DIVERGENT = (
    "training.py",
    "train.py",
    "phase2_verify.py",
    # --from-jsonl replay + published-artifact support added release-side for
    # the CI "verify published rate artifacts" step. A sync with a stale
    # DIVERGENT list clobbered these once and broke CI — if upstream changes
    # them, port by hand.
    "phase2_temperature.py",
    "phase2_decay.py",
    "phase2_interp.py",
)

# Applied after the mechanical rewrite. Each `old` MUST be found, or we abort —
# a missing anchor means upstream moved and the patch needs re-deriving.
# ruff: noqa: E501 — the anchors below are verbatim upstream lines; reflowing
# them would stop them matching.
PATCHES: dict[str, list[tuple[str, str]]] = {
    "figures.py": [
        (
            '    figdir = Path(__file__).parent / "figures"',
            '    figdir = Path(__file__).resolve().parent.parent / "figures"',
        ),
    ],
    "config.py": [
        # CLAUDE.md is monorepo-only and does not ship here.
        (
            "  settings so callers get good results without overrides (per CLAUDE.md).",
            "  settings so callers get good results without overrides.",
        ),
        (
            "    # --- Streaming data (always single-epoch unique data; see CLAUDE.md) ---",
            "    # --- Streaming data (always single-epoch unique data) ---",
        ),
    ],
    "analysis.py": [
        (
            'resolve_checkpoint(checkpoint, Path("data/retok/ckpt_cache"))',
            "resolve_checkpoint(checkpoint)",
        ),
        # The private bucket has no public equivalent; point at the HF dataset.
        (
            "        --checkpoint s3://brendanlong-experiments/retok/checkpoints/<run>/final.pt \\",
            "        --checkpoint hf:checkpoints/retok-main-s0/final.pt \\",
        ),
        (
            'parser.add_argument("--checkpoint", required=True, help="local path or s3:// URI")',
            "parser.add_argument(\n"
            '        "--checkpoint",\n'
            "        required=True,\n"
            "        help=(\n"
            '            "Local path, or hf:<relpath> into the published dataset "\n'
            '            "(e.g. hf:checkpoints/retok-main-s0/final.pt)."\n'
            "        ),\n"
            "    )",
        ),
        (
            "    if len(set(y.tolist())) < 2:\n        return float((y == y[0]).mean())",
            "    if len(set(y.tolist())) < 2:\n"
            "        # Single-class bucket: a constant predictor is trivially perfect.\n"
            "        return 1.0",
        ),
        (
            "import torch\nfrom sklearn.linear_model import LogisticRegression",
            "import numpy as np\nimport torch\n"
            "from sklearn.linear_model import LogisticRegression",
        ),
        (
            "    return float((clf.predict(x[te]) == y[te]).mean())",
            "    preds = np.asarray(clf.predict(x[te]))\n"
            "    return float(np.mean(preds == np.asarray(y[te])))",
        ),
    ],
}

# Checked after every sync. These encode the strips that hand-syncs kept undoing.
INVARIANTS: list[tuple[str, str, bool]] = [
    # (description, needle, should_be_present)
    ("no S3 upload in training", "upload_checkpoint_to_s3", False),
    ("no run-name guards in train", "assert_run_name_free", False),
    ("no monorepo-only CLAUDE.md refs", "CLAUDE.md", False),
    ("hf: checkpoint resolution documented", "hf:checkpoints/", True),
    ("verifier keeps --all-published", "--all-published", True),
]


def rewrite_imports(text: str) -> str:
    text = text.replace(f"experiments.{EXPERIMENT}", EXPERIMENT)
    for module in SHARED_MODULES:
        text = text.replace(f"shared.{module}", f"common.{module}")
    return text


def sync_code(src: Path, dest: Path) -> list[str]:
    copied, skipped = [], []
    for path in sorted(src.glob("*.py")):
        if path.name in DIVERGENT:
            skipped.append(path.name)
            continue
        text = rewrite_imports(path.read_text())
        for old, new in PATCHES.get(path.name, []):
            if old not in text:
                sys.exit(
                    f"ABORT: patch anchor missing in {path.name}:\n  {old[:90]!r}\n"
                    "Upstream changed the code this patch targets. Re-derive it in "
                    "PATCHES rather than syncing a release that silently lost a strip."
                )
            text = text.replace(old, new)
        (dest / EXPERIMENT / path.name).write_text(text)
        copied.append(path.name)
    for path in sorted((src / "tests").glob("*.py")):
        (dest / EXPERIMENT / "tests" / path.name).write_text(
            rewrite_imports(path.read_text())
        )
        copied.append(f"tests/{path.name}")
    if skipped:
        print(f">>> NOT synced (release-only divergence): {', '.join(skipped)}")
        print("    If upstream changed these, port the change by hand.")
    return copied


def sync_docs(src: Path, dest: Path) -> None:
    # WRITEUP: flatten paths, and keep the release's reproduction snippet.
    writeup = rewrite_imports(src.joinpath("WRITEUP.md").read_text())
    writeup = writeup.replace(
        f"*Draft — August 2026. Code and data: `experiments/{EXPERIMENT}/`.",
        f"*Draft — August 2026. Code: `{EXPERIMENT}/`.",
    )
    dest.joinpath("WRITEUP.md").write_text(writeup)

    # RESULTS ships verbatim apart from module paths, and keeps its header note.
    # The s3:// URIs must survive: they are the historical record, and the note
    # maps them onto this repo.
    results = src.joinpath("RESULTS.md").read_text()
    results = results.replace(f"experiments.{EXPERIMENT}", EXPERIMENT).replace(
        f"`experiments/{EXPERIMENT}/`", f"`{EXPERIMENT}/`"
    )
    if "s3://brendanlong-experiments/" not in results:
        sys.exit("ABORT: RESULTS.md lost its s3:// URIs — the rewrite is too greedy.")
    current = dest.joinpath("RESULTS.md").read_text()
    note = current[
        current.index("> **Note on this log.") : current.index("\n\n**Question.**")
    ]
    title = "# Results: retok (hidden computational composition)\n"
    results = results.replace(title, title + "\n" + note + "\n", 1)
    dest.joinpath("RESULTS.md").write_text(results)

    plan = rewrite_imports(src.joinpath("EXPERIMENT_PLAN.md").read_text())
    dest.joinpath("EXPERIMENT_PLAN.md").write_text(plan)

    for fig in sorted((src / "figures").glob("*.png")):
        shutil.copy(fig, dest / "figures" / fig.name)


def _skip(path: Path, dest: Path) -> bool:
    """This script names the forbidden needles in its own config, so exempt it."""
    return ".venv" in path.parts or path.resolve() == Path(__file__).resolve()


def check_invariants(dest: Path) -> None:
    failures = []
    for desc, needle, want in INVARIANTS:
        hits = [
            p
            for p in dest.rglob("*.py")
            if not _skip(p, dest) and needle in p.read_text()
        ]
        if want and not hits:
            failures.append(f"  MISSING: {desc} ({needle!r} found nowhere)")
        if not want and hits:
            names = ", ".join(str(p.relative_to(dest)) for p in hits)
            failures.append(f"  PRESENT: {desc} ({needle!r} in {names})")
    # s3:// belongs only in RESULTS.md (historical) and the two files that
    # explain the mapping.
    allowed = {"RESULTS.md", "checkpoint.py", "artifacts.py"}
    for path in dest.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".sh"}:
            if _skip(path, dest) or path.name in allowed:
                continue
            if "s3://" in path.read_text():
                failures.append(f"  PRESENT: stray s3:// in {path.relative_to(dest)}")
    if failures:
        sys.exit("ABORT: release invariants violated:\n" + "\n".join(failures))
    print(">>> invariants OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monorepo", required=True, help="Path to the monorepo checkout"
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    src = Path(args.monorepo).expanduser() / "experiments" / EXPERIMENT
    dest = Path(__file__).resolve().parent.parent
    if not src.is_dir():
        sys.exit(f"ABORT: no experiment at {src}")

    if not args.check_only:
        copied = sync_code(src, dest)
        sync_docs(src, dest)
        print(f">>> synced {len(copied)} code files + docs + figures")

    if not args.check_only:
        # The import rewrite changes sort order (common < retok), so formatting
        # is part of the transform, not a follow-up chore.
        for cmd in (
            ["ruff", "check", "--fix", "-q", "."],
            ["ruff", "format", "-q", "."],
        ):
            subprocess.run(["uv", "run", *cmd], cwd=dest, check=False)
        print(">>> ruff check --fix + format applied")

    check_invariants(dest)
    print(">>> now run: uv run pyright && uv run pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
