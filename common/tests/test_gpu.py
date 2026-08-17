"""Tests for the GPU preflight helper.

These run on both CPU-only CI and GPU hosts, so availability-dependent
assertions branch on ``torch.cuda.is_available()``. The env-var parsing and the
"never fall back silently" contract are tested deterministically.
"""

import pytest
import torch

from common import gpu


def test_env_requires_cuda_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ["1", "true", "TRUE", "yes", "on"]:
        monkeypatch.setenv("REQUIRE_CUDA", val)
        assert gpu._env_requires_cuda() is True
    for val in ["0", "false", "no", "", "off"]:
        monkeypatch.setenv("REQUIRE_CUDA", val)
        assert gpu._env_requires_cuda() is False
    monkeypatch.delenv("REQUIRE_CUDA", raising=False)
    assert gpu._env_requires_cuda() is False


def test_describe_gpu_is_a_string() -> None:
    desc = gpu.describe_gpu()
    assert "torch=" in desc
    assert ("cuda_available=False" in desc) == (not torch.cuda.is_available())


def test_assert_matches_availability() -> None:
    if torch.cuda.is_available():
        gpu.assert_cuda_healthy()  # should not raise on a healthy GPU host
    else:
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            gpu.assert_cuda_healthy()


def test_resolve_device_require_true_matches_availability() -> None:
    if torch.cuda.is_available():
        assert gpu.resolve_device(require_cuda=True).type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            gpu.resolve_device(require_cuda=True)


def test_resolve_device_require_false_never_raises() -> None:
    # Explicitly not required: always returns a device, never raises, even
    # when CUDA is missing (falls back to CPU with a warning).
    dev = gpu.resolve_device(require_cuda=False)
    assert dev.type in {"cuda", "cpu"}
    if not torch.cuda.is_available():
        assert dev.type == "cpu"


def test_assert_catches_broken_but_available_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core case the PR defends against: is_available() is True but a real
    GPU op raises ("driver too old"). Simulated so it runs on CPU CI too."""

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("CUDA driver version is insufficient (driver too old)")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu, "describe_gpu", lambda: "stub-desc")
    monkeypatch.setattr(torch, "zeros", boom)
    with pytest.raises(RuntimeError, match="GPU op failed"):
        gpu.assert_cuda_healthy()
    # resolve_device(require_cuda=False) must NOT hand back the broken device:
    # it falls back to CPU instead of a device that would blow up mid-training.
    assert gpu.resolve_device(require_cuda=False).type == "cpu"


def test_resolve_device_none_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # require_cuda=None defers to REQUIRE_CUDA. On a CPU-only host, REQUIRE_CUDA=1
    # must turn the silent CPU fallback into a hard failure.
    if not torch.cuda.is_available():
        monkeypatch.setenv("REQUIRE_CUDA", "1")
        with pytest.raises(RuntimeError):
            gpu.resolve_device(require_cuda=None)
        monkeypatch.setenv("REQUIRE_CUDA", "0")
        assert gpu.resolve_device(require_cuda=None).type == "cpu"
