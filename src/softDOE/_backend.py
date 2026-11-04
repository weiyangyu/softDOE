"""Native Metal / optional PyTorch CUDA discovery and CPU fallback."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import platform
import numpy as np

from ._engine import ArrayOps, QState


def _entry(compiled=False, available=False, device=None, reason=None) -> dict:
    return dict(compiled=compiled, available=available, device=device, reason=reason)


@lru_cache(maxsize=1)
def _capabilities() -> dict:
    info = {"cpu": _entry(True, True, "CPU"),
            "metal": _entry(reason="Metal is supported only on macOS."),
            "cuda": _entry(reason="CUDA is supported only on Linux.")}
    system = platform.system()
    target = "metal" if system == "Darwin" else "cuda" if system == "Linux" else None
    if target is None:
        return info
    if target == "metal":
        try:
            from ._metal import capabilities
            info[target] = capabilities()
        except Exception as error:
            info[target] = _entry(reason=f"Native Metal unavailable: {error}")
        return info
    try:
        import torch
    except Exception as error:
        info[target] = _entry(reason=f"Optional PyTorch GPU runtime unavailable: {error}")
        return info
    try:
        compiled = torch.version.cuda is not None
        available = torch.cuda.is_available()
        device = torch.cuda.get_device_name() if available else None
        info[target] = _entry(compiled=compiled)
        if not available:
            info[target]["reason"] = f"No usable {target} GPU is available in this process."
            return info
        # Exercise the actual correlation, Cholesky, triangular solve and append
        # path; successful import/device discovery alone is not enough.
        ops = ArrayOps(target)
        state = QState(np.array([[0.1, 0.2], [0.8, 0.9]]),
                       np.array([[0.25, 0.75], [0.5, 0.5]]), 0.1, ops)
        state = state.append(np.array([[0.4, 0.6]]))
        if not np.isfinite(state.score):
            raise RuntimeError("GPU numerical self-test failed.")
        reference = QState(ops.numpy(state.design), ops.numpy(state.evaluation),
                           0.1, ArrayOps()).score
        if not np.isclose(state.score, reference, rtol=1e-4, atol=1e-5):
            raise RuntimeError("GPU numerical self-test disagrees with CPU.")
        from ._cuda import PairedUpdates
        blocks = (np.array([[0.3, 0.7]]), np.array([[0.6, 0.2]]))
        updates = PairedUpdates(state, *blocks)
        expected = [QState(np.vstack((ops.numpy(state.design), block)),
                           ops.numpy(state.evaluation), 0.1, ArrayOps()).score
                    for block in blocks]
        if not np.allclose(updates.scores, expected, rtol=1e-4, atol=1e-5):
            raise RuntimeError("Paired CUDA numerical self-test disagrees with CPU.")
        info[target] = _entry(compiled, True, device)
    except Exception as error:
        info[target]["reason"] = f"{target} runtime self-test failed: {error}"
    return info


def backend_info(refresh: bool = False) -> dict:
    """Report compiled/available/device/reason for cpu, metal, and cuda.

    Metal uses the native macOS runtime. CUDA on Linux uses optional PyTorch
    (``pip install 'softDOE[gpu]'``). A numerical self-test must pass before
    a GPU is reported as available. ``compiled`` describes support in the
    native extension (Metal) or PyTorch runtime (CUDA). CPU needs no PyTorch.
    Set ``refresh=True`` to repeat discovery after installing a runtime.
    """
    if not isinstance(refresh, bool):
        raise TypeError("refresh must be True or False.")
    if refresh:
        _capabilities.cache_clear()
    return deepcopy(_capabilities())


def _resolve_backend(backend) -> tuple:
    requested = backend.lower() if isinstance(backend, str) else ""
    if requested not in ("auto", "cpu", "cuda", "metal"):
        return "cpu", "CPU", "The requested backend is invalid."
    if requested == "cpu":
        return "cpu", "CPU", None
    capabilities = backend_info()
    if requested != "auto":
        selected = capabilities[requested]
        if not selected["available"]:
            return "cpu", "CPU", selected["reason"]
        return requested, selected["device"], None
    preferred = "metal" if platform.system() == "Darwin" else "cuda"
    if capabilities[preferred]["available"]:
        return preferred, capabilities[preferred]["device"], None
    return "cpu", "CPU", None


def _run_with_fallback(backend, workflow) -> tuple:
    selected, device, reason = _resolve_backend(backend)
    if selected == "cpu":
        return workflow(ArrayOps()), "cpu", "CPU", reason
    try:
        if selected == "metal":
            from ._metal import MetalOps
            ops = MetalOps()
        else:
            ops = ArrayOps(selected)
        return workflow(ops), selected, device, reason
    except Exception as error:
        stage = getattr(error, "soft_stage", "design generation")
        reason = f"{selected} backend execution failed during {stage}: {error}"
        # Re-run from original inputs, never keep a partially optimized GPU base.
        return workflow(ArrayOps()), "cpu", "CPU", reason
