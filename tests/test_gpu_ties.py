"""MOFAT's GPU tie policy, tested without requiring either GPU device."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from softDOE import _soft
from softDOE._engine import ArrayOps, cpu_score, gpu_close
from softDOE._strict_metal import StrictMetalEvaluator


@pytest.mark.parametrize("backend", ["cpu", "cuda", "metal"])
@pytest.mark.parametrize("structure", ["standard", "strict"])
@pytest.mark.parametrize("margin", [-0.5, 0, 0.5, 2])
def test_swap_keeps_gpu_incumbent_on_ties(monkeypatch, backend, structure, margin):
    ops = SimpleNamespace(backend=backend)
    A = np.array([[0.1, 0.2], [0.7, 0.8]])
    prepared = dict(A=A, p=2, l=2, qmc=A, beta=1/16,
                    remainder=None, quotient=None)
    old = -100.0
    new = old + margin * np.sqrt(np.finfo(np.float32).eps) * abs(old)
    calls = []
    def initial(candidate, *args):
        calls.append(candidate.copy())
        return candidate.copy(), old if len(calls) == 1 else new
    def forbidden(*args):
        raise AssertionError("GPU ties must not be rescored on CPU")
    monkeypatch.setattr(_soft, "_initial", initial)
    monkeypatch.setattr(_soft, "cpu_score", forbidden, raising=False)
    monkeypatch.setattr(StrictMetalEvaluator, "__init__", lambda self, *args: None)
    design, score = _soft._optimize(prepared, structure, ops)
    accept = margin > (0 if backend == "cpu" else 1)
    assert len(calls) == 2
    np.testing.assert_array_equal(design, calls[1 if accept else 0])
    assert score == (new if accept else old)


@pytest.mark.parametrize("backend,fused", [
    ("cpu", False), ("cuda", False), ("metal", False), ("metal", True)])
@pytest.mark.parametrize("margin", [-0.5, 0, 0.5, 2])
def test_strict_construction_prefers_left_on_gpu_ties(monkeypatch, backend, fused, margin):
    ops = SimpleNamespace(backend=backend)
    A = np.array([[0.1, 0.2], [0.7, 0.8]])
    prepared = dict(p=2, delta=0.4, beta=1/16, qmc=A)
    left = -100.0
    right = left + margin * np.sqrt(np.finfo(np.float32).eps) * abs(left)
    class State:
        def __init__(self, *args):
            self.index = 0
        def append(self, block):
            self.index += 1
            return SimpleNamespace(score=left if self.index == 1 else right)
    class Fused:
        def initialize(self, design):
            pass
        def evaluate_pair(self, a, b):
            return [left, right]
        def accept(self, index):
            self.selected = index
    def forbidden(*args):
        raise AssertionError("GPU ties must not be rescored on CPU")
    monkeypatch.setattr(_soft, "QState", State)
    monkeypatch.setattr(_soft, "cpu_score", forbidden, raising=False)
    evaluator = Fused() if fused else None
    design, score = _soft._initial(A, prepared, "strict", ops, evaluator)
    initial_right = _soft._shift_factor(A, 0, prepared["delta"])
    choose_right = margin > (0 if backend == "cpu" else 1)
    if choose_right:
        expected = np.vstack((A, initial_right,
                              _soft._shift_factor(initial_right, 1, prepared["delta"])))
    else:
        expected = np.vstack((_soft._shift_factor(A, 1, prepared["delta"]), A, initial_right))
    np.testing.assert_array_equal(design, expected)
    assert score == (right if choose_right else left)


def test_cuda_and_metal_use_the_same_mofat_tolerance():
    # This difference is close by MOFAT's float32 tolerance, but not float64's.
    for backend in ("cuda", "metal"):
        assert gpu_close(-100.0, -99.99, SimpleNamespace(backend=backend))
        assert not gpu_close(-100.0, -99.9, SimpleNamespace(backend=backend))


def test_standard_matches_mofat_metal(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("No Apple Metal device available")
    reference = json.loads((Path(__file__).parent / "fixtures/mofat_standard_metal.json").read_text())
    p, l, m = (reference[key] for key in ("p", "l", "m"))
    prepared = _soft._prepare(p, l, m, 123, 2 * l)
    prepared["A"] = np.asarray(reference["A"])
    def forbidden(*args):
        raise AssertionError("Standard Metal must not rescore ties on CPU")
    monkeypatch.setattr(_soft, "cpu_score", forbidden, raising=False)
    design, score = _soft._optimize(prepared, "standard", ArrayOps("metal"))
    np.testing.assert_allclose(design, reference["design"], rtol=0, atol=2e-14)
    np.testing.assert_allclose(score, reference["log_Q"], rtol=0, atol=3e-5)
    np.testing.assert_allclose(score, cpu_score(design, prepared["qmc"], prepared["beta"]),
                               rtol=0, atol=3e-5)
