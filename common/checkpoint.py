"""Checkpoint save/load utilities.

Training writes checkpoints locally; published checkpoints are hosted in a
public HuggingFace dataset (see :mod:`common.artifacts`). Use
:func:`resolve_checkpoint` wherever a checkpoint is a CLI argument — it accepts
a local path or an ``hf:<relpath>`` reference into that dataset.

The monorepo this was extracted from uploads checkpoints to a private S3 bucket;
that path has no public equivalent and was removed. ``s3://`` URIs appearing in
RESULTS.md map onto ``hf:checkpoints/<run_name>/<filename>``.
"""

from pathlib import Path

import torch

from common.artifacts import artifact_path


def save_model_checkpoint(
    model: torch.nn.Module,
    step: int,
    model_config: dict[str, object],
    checkpoint_dir: Path,
    filename: str | None = None,
) -> Path:
    """Save a checkpoint, stripping any torch.compile ``_orig_mod.`` prefix.

    The payload is ``{"step", "model_state_dict", "model_config"}`` so
    loaders can reconstruct the model from its config dict. Loadable with
    ``torch.load(..., weights_only=True)``.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()
    }
    ckpt_path = checkpoint_dir / (filename or f"step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "model_config": model_config,
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")
    return ckpt_path


def resolve_checkpoint(source: str, cache_dir: Path | None = None) -> Path:
    """Resolve a checkpoint source to a local path.

    - A local path is returned as-is.
    - ``hf:<relpath>`` is downloaded from the public dataset (HuggingFace
      manages its own cache; ``cache_dir`` is accepted and ignored so callers
      that thread one through keep working).
    """
    del cache_dir
    if source.startswith("hf:"):
        return Path(artifact_path(source.removeprefix("hf:")))
    return Path(source)
