"""GPU preflight: fail fast when a cloud host can't actually run CUDA.

The most expensive RunPod/SkyPilot failure mode we see is *silent CPU
degradation*: the job shows RUNNING and healthy, but ``torch.cuda.is_available()``
is ``False`` (or a CUDA op raises "driver too old"), so training crawls on CPU
for hours before anyone notices. This happens when the torch CUDA build doesn't
match the host driver (e.g. a cu13 wheel on a driver-570/CUDA-12.8 host), or when
an eval sub-venv was built with a CPU-only torch.

``assert_cuda_healthy`` turns that hours-long silent burn into a ~2-second, loud
failure: it checks availability *and* runs a real GPU op (which is what actually
catches the driver-mismatch case, where ``nvidia-smi`` succeeds but torch can't
initialize CUDA). Run it as the first step of every cloud ``run:`` block via
``python -m common.gpu --require-cuda``; it covers every downstream invocation in
that job. Re-place onto a healthy host rather than retrying in place.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_TRUTHY = {"1", "true", "yes", "on"}


def _env_requires_cuda() -> bool:
    """Whether the REQUIRE_CUDA env var is set to a truthy value."""
    return os.environ.get("REQUIRE_CUDA", "").strip().lower() in _TRUTHY


def describe_gpu() -> str:
    """One-line summary of the torch/CUDA/driver situation, for logging."""
    parts = [f"torch={torch.__version__}", f"torch_cuda_build={torch.version.cuda}"]
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        parts.append(f"device={torch.cuda.get_device_name(0)}")
        parts.append(f"capability=sm_{cap[0]}{cap[1]}")
    else:
        parts.append("cuda_available=False")
    # The host driver's CUDA version (the usual mismatch culprit) is reported by
    # the separate `nvidia-smi` line in the YAML run block, not exposed by torch.
    return " ".join(parts)


def assert_cuda_healthy() -> None:
    """Raise a clear error unless CUDA is available *and* a real GPU op works.

    Catches both failure shapes of the driver/torch-build mismatch:
      * ``torch.cuda.is_available()`` is ``False`` (the common case — torch built
        for a newer CUDA than the host driver supports), and
      * it returns ``True`` but the first real op raises "driver too old".
    """
    hint = (
        "This host cannot run CUDA with the installed torch. Almost always a "
        "torch-build vs host-driver mismatch (e.g. a cu13 wheel on a driver-570 "
        "/ CUDA-12.8 host) or a CPU-only torch in this venv. Pin the torch CUDA "
        "build to match the fleet driver (see your cloud provider's docs and the "
        "host, not this job), or re-place onto a healthy host."
    )
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available. {describe_gpu()}. {hint}")
    try:
        # A real allocation + kernel launch + host sync; this is what trips the
        # "driver too old" error that is_available() sometimes misses.
        probe = torch.zeros(8, device="cuda")
        _ = (probe + 1.0).sum().item()
    except Exception as exc:  # RuntimeError "driver too old", OSError, etc.
        raise RuntimeError(
            f"CUDA is reported available but a GPU op failed: {exc}. "
            f"{describe_gpu()}. {hint}"
        ) from exc


def resolve_device(require_cuda: bool | None = None) -> torch.device:
    """Return the training device, failing fast when CUDA is required but broken.

    Args:
        require_cuda: If ``True``, assert a working GPU and never fall back to
            CPU. If ``None`` (default), require CUDA iff the ``REQUIRE_CUDA`` env
            var is truthy — cloud ``run:`` blocks set it, so real runs fail fast
            while local CPU smoke tests still work. If ``False``, use CUDA when
            healthy and otherwise fall back to CPU with a loud warning (so the
            fallback is never silent).
    """
    if require_cuda is None:
        require_cuda = _env_requires_cuda()
    if require_cuda:
        assert_cuda_healthy()
        return torch.device("cuda")
    # Not required: prefer CUDA, but only if it actually works. On an
    # available-but-broken host, fall back to CPU rather than hand back a device
    # that blows up mid-training. Either way the fallback is loud, never silent.
    reason = ""
    if torch.cuda.is_available():
        try:
            assert_cuda_healthy()
            return torch.device("cuda")
        except RuntimeError as exc:
            reason = f" ({exc})"
    print(
        f">>> WARNING: CUDA unavailable/unhealthy, falling back to CPU "
        f"({describe_gpu()}){reason}. Set REQUIRE_CUDA=1 to make this a hard "
        "failure on GPU hosts.",
        file=sys.stderr,
    )
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU preflight check.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero unless a working CUDA GPU is present.",
    )
    args = parser.parse_args()
    print(f">>> GPU preflight: {describe_gpu()}")
    if args.require_cuda or _env_requires_cuda():
        try:
            assert_cuda_healthy()
        except RuntimeError as exc:
            print(f">>> GPU preflight FAILED: {exc}", file=sys.stderr)
            return 1
        print(">>> GPU preflight OK: CUDA is available and a probe op succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
