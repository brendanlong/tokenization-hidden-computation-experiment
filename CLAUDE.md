# Working conventions for this repo

Public release repo for the retok experiments (extracted from the private
`brendanlong/experiments` monorepo — broader conventions live in that repo's
CLAUDE.md; this file carries only what is specific to working here).

## Measuring tokenization — the rules that have bitten us

- **The canonicality metric is the round trip**: a generation is
  non-canonical iff `encode(decode(ids)) != ids` on the emitted token IDs.
  An unexpected-but-canonical token is NOT a non-canonical token: text that
  is simply wrong (junk, wrong casing) contributes zero to segmentation
  metrics — its only legitimate effect is on correctness/reward metrics.
- **Per-token rates are primary.** Per-generation flags grow mechanically
  with output length (BPE is non-recovering), so a sequence-level rate is
  uninterpretable without its length. Report both, per-token first.
- Segmentation-of-the-answer classifications (canonical / greedy-longest /
  all-single-char) are only defined for compliant outputs, compared against
  the **case-preserved** produced surface. The primary form is the
  **correct-region** family: per-token metrics computed only over tokens
  lying entirely within the correct leading prefix of the answer (wrong
  characters excluded by construction), optionally broken out per character
  position. See `retok_rl/reversal.py::summarise_reversal` for the
  reference implementation of all metric families.
- Exclude U+FFFD round trips; NFC-normalize concerns apply for Qwen; pin
  sampling params and dtype (see WRITEUP.md "If you measure this yourself").

## Writing up results

Report what was run and which metrics changed and by how much — tables of
trajectories with one factual reading note each, prediction status stated
against the pre-registration. Keep interpretation minimal and clearly
separated; overclaiming has required public retractions here twice.

- Root `RESULTS.md` is a **verbatim** provenance log: never edit the body;
  add dated erratum annotations to the block at the top.
- `retok_rl/RESULTS.md` is a living run log: corrections go in-place with a
  dated correction note, superseded numbers explicitly retracted.
- Pre-register predictions in the relevant EXPERIMENT_PLAN.md *before*
  launching a run; post-run changes are dated annotations.

## Reproduction / CI

- `uv sync`; CI = `ruff check`, `ruff format --check`, `pyright`,
  `pytest -m "not smoke"`, plus artifact verification — run all locally
  before pushing.
- Analyses need no GPU: `uv run python -m retok.phase2_verify
  --all-published` and `bash scripts/reproduce_analyses.sh`.
- GPU runs go through SkyPilot (`skypilot/*.yaml`) with auto-down; the pod
  venv needs the cu128 torch swap and `uv run --no-sync` (see
  `skypilot/train-retok-rl.yaml` comments). Artifacts: rsync via the
  cluster ssh config, with the gzip+base64 log escape hatch
  (`scripts/extract_rollouts_from_log.sh`) as backup.
- Keep exploratory GPU spend ~$1–5 per run; ask before anything ≥$10,
  stating the price.

## Publishing

Generation artifacts and checkpoints live in the HF dataset
`brendanlong/retok-noncanonical-tokenization`; uploads need a write token
(the ambient token on brendan-desktop is read-only). The dataset card is
`hf_dataset_card.md` (repo copy) → dataset `README.md`.
