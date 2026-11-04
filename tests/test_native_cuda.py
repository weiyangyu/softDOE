"""Actual native CUDA search, compared with the independent tensor path."""
import numpy as np
import pytest

import softDOE
from softDOE import _native
from softDOE._cuda import PairedUpdates
from softDOE._cuda_native import configuration
from softDOE._engine import ArrayOps
from softDOE._soft import _prepare, _optimize


@pytest.fixture
def native_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not hasattr(_native, "optimize_cuda"):
        pytest.skip("No native CUDA device/build")
    config = configuration(torch.cuda.current_device())
    if config is None:
        pytest.skip("NVRTC is unavailable")
    return config


@pytest.mark.parametrize("p,l,m", [(3,2,64), (4,4,64), (20,6,500)])
def test_native_strict_matches_tensor_search(native_cuda, p, l, m, monkeypatch):
    prepared = _prepare(p,l,m,123,2*l)
    actual, score = _optimize(prepared,"strict",ArrayOps("cuda"))
    monkeypatch.setattr("softDOE._cuda_native.configuration",lambda device:None)
    expected, reference = _optimize(prepared,"strict",ArrayOps("cuda"))
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_allclose(score,reference,rtol=2e-12,atol=2e-12)


def test_public_strict_uses_compiled_loop(native_cuda, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native strict search must not use Python paired updates")
    # Complete the shared CUDA numerical self-test before blocking its tensor probe.
    assert softDOE.backend_info(refresh=True)["cuda"]["available"]
    monkeypatch.setattr(PairedUpdates,"__init__",forbidden)
    result=softDOE.soft(4,4,"strict",m=32,seed=123,backend="cuda")
    assert result["backend_used"]=="cuda",result["fallback_reason"]


def test_strict_cuda_preserves_underflow_normalization(native_cuda):
    # These distant evaluation points force the product path below float64's
    # normal range. Compare the normalized score with a full CPU solve.
    prepared = _prepare(100,2,8,123,4)
    prepared["qmc"] = np.full((3,100), 20.0)
    design, log_q = _optimize(prepared,"strict",ArrayOps("cuda"))
    expected = softDOE.Q(1,prepared["beta"],design,prepared["qmc"],"cpu")
    assert expected["Q"] == 0
    assert expected["normalized_Q"] > 0
    np.testing.assert_allclose(np.exp(log_q/100),expected["normalized_Q"],rtol=2e-12)


def test_invalid_native_kernel_falls_back_to_cpu(native_cuda, monkeypatch):
    source,library,device,arch=native_cuda
    monkeypatch.setattr("softDOE._cuda_native.configuration",lambda _: ("invalid CUDA source",library,device,arch))
    result=softDOE.soft(3,2,"strict",m=32,backend="cuda")
    expected=softDOE.soft(3,2,"strict",m=32,backend="cpu")
    assert result["backend_used"]=="cpu"
    assert "compilation failed" in result["fallback_reason"]
    np.testing.assert_array_equal(result["design"],expected["design"])
