"""Tests for common.checkpoint (no network: the HF download is mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import torch

from common import checkpoint


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


def test_save_model_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _Tiny()
    path = checkpoint.save_model_checkpoint(
        model, step=7, model_config={"dim": 2}, checkpoint_dir=tmp_path
    )
    assert path == tmp_path / "step_7.pt"
    loaded = torch.load(path, weights_only=True)
    assert loaded["step"] == 7
    assert loaded["model_config"] == {"dim": 2}
    assert set(loaded["model_state_dict"]) == {"lin.weight", "lin.bias"}


def test_save_model_checkpoint_strips_compile_prefix(tmp_path: Path) -> None:
    """torch.compile wraps params under ``_orig_mod.``; checkpoints must not."""
    model = _Tiny()
    with mock.patch.object(
        model,
        "state_dict",
        return_value={f"_orig_mod.{k}": v for k, v in model.state_dict().items()},
    ):
        path = checkpoint.save_model_checkpoint(
            model, step=1, model_config={}, checkpoint_dir=tmp_path
        )
    keys = torch.load(path, weights_only=True)["model_state_dict"]
    assert set(keys) == {"lin.weight", "lin.bias"}
    assert not any(k.startswith("_orig_mod.") for k in keys)


def test_resolve_checkpoint_local_path_passthrough(tmp_path: Path) -> None:
    local = tmp_path / "final.pt"
    local.touch()
    assert checkpoint.resolve_checkpoint(str(local)) == local


def test_resolve_checkpoint_hf_prefix_downloads() -> None:
    """``hf:<relpath>`` resolves through the public dataset, stripping the prefix."""
    with mock.patch.object(
        checkpoint, "artifact_path", return_value="/cache/final.pt"
    ) as dl:
        out = checkpoint.resolve_checkpoint("hf:checkpoints/retok-main-s0/final.pt")
    dl.assert_called_once_with("checkpoints/retok-main-s0/final.pt")
    assert out == Path("/cache/final.pt")
