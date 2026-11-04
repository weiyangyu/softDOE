"""Actual Metal regression tests, including the reported 20-factor slowdown."""
import json
from pathlib import Path

import numpy as np
import pytest

import softDOE
from softDOE._engine import ArrayOps, cpu_score
from softDOE._soft import _prepare, _optimize
from softDOE._strict_metal import StrictMetalEvaluator


@pytest.fixture
def metal_ops():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("No Apple Metal device available")
    # Do not skip on a failed softDOE self-test: shader failures must fail here.
    assert softDOE.backend_info()["metal"]["available"]
    return ArrayOps("metal")


@pytest.mark.parametrize("p,k,m", [(2, 1, 17), (20, 6, 33), (100, 2, 7)])
def test_fused_pairs_and_accepted_updates(p, k, m, metal_ops):
    rng = np.random.default_rng(42)
    design, points = rng.random((2 * k, p)), rng.random((m, p))
    beta = 1 / (2 * k)**2
    evaluator = StrictMetalEvaluator(points, beta, k, 5 * k, metal_ops)
    evaluator.initialize(design)
    for choice in (0, 1, 0):
        candidates = [rng.random((k, p)), rng.random((k, p))]
        actual = evaluator.evaluate_pair(*candidates)
        expected = [cpu_score(np.vstack((design, block)), points, beta)
                    for block in candidates]
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-5)
        evaluator.accept(choice)
        design = np.vstack((design, candidates[choice]))
    # Reusing the workspace must not retain the preceding design's state.
    evaluator.initialize(design[:2 * k])
    actual = evaluator.evaluate_pair(*candidates)
    expected = [cpu_score(np.vstack((design[:2 * k], block)), points, beta)
                for block in candidates]
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-5)


def test_fused_log_scaling_survives_correlation_underflow(metal_ops):
    points = np.ones((7, 100))
    design = np.zeros((4, 100))
    block = np.full((2, 100), 0.1)
    evaluator = StrictMetalEvaluator(points, 1 / 144, 2, 6, metal_ops)
    evaluator.initialize(design)
    actual = evaluator.evaluate_pair(block, block)
    expected = cpu_score(np.vstack((design, block)), points, 1 / 144)
    assert expected < -745  # Even an unscaled float64 Q underflows here.
    np.testing.assert_allclose(actual, expected, rtol=3e-6)


def test_strict_20_6_matches_mofat_metal(metal_ops, monkeypatch):
    reference = json.loads((Path(__file__).parent / "fixtures/mofat_strict_metal.json").read_text())
    prepared = _prepare(reference["p"], reference["l"], reference["m"], 123, 12)
    prepared["A"] = np.asarray(reference["A"])

    def forbidden(*args):
        raise AssertionError("Strict Metal must not recompute close candidates on CPU")
    monkeypatch.setattr("softDOE._soft.cpu_score", forbidden, raising=False)
    design, score = _optimize(prepared, "strict", metal_ops)
    np.testing.assert_allclose(design, reference["metal"]["design"], rtol=0, atol=2e-14)
    np.testing.assert_allclose(score, reference["metal"]["log_Q"], rtol=0, atol=3e-5)
    # Independently verify the GPU-reported score for its chosen design.
    np.testing.assert_allclose(score, cpu_score(design, prepared["qmc"], prepared["beta"]),
                               rtol=0, atol=3e-5)


def test_strict_20_6_matches_mofat_cpu():
    reference = json.loads((Path(__file__).parent / "fixtures/mofat_strict_metal.json").read_text())
    prepared = _prepare(reference["p"], reference["l"], reference["m"], 123, 12)
    prepared["A"] = np.asarray(reference["A"])
    design, score = _optimize(prepared, "strict", ArrayOps())
    np.testing.assert_allclose(design, reference["cpu"]["design"], rtol=0, atol=2e-14)
    np.testing.assert_allclose(score, reference["cpu"]["log_Q"], rtol=0, atol=2e-12)


def test_strict_kernel_failure_restarts_on_cpu(monkeypatch):
    from softDOE import _backend
    expected = softDOE.soft(3, 2, "strict", m=16, backend="cpu")
    monkeypatch.setattr(_backend, "_resolve_backend", lambda b: ("metal", "Test GPU", None))
    class SimulatedOps(ArrayOps):
        def __init__(self, backend="cpu"):
            super().__init__()
            self.backend = backend
    monkeypatch.setattr(_backend, "ArrayOps", SimulatedOps)
    def fail(*args, **kwargs):
        raise RuntimeError("simulated Metal compilation failure")
    from softDOE import _native
    monkeypatch.setattr(_native, "optimize_metal", fail, raising=False)
    actual = softDOE.soft(3, 2, "strict", m=16, backend="metal")
    assert actual["backend_used"] == "cpu"
    assert "simulated Metal compilation failure" in actual["fallback_reason"]
    np.testing.assert_array_equal(actual["design"], expected["design"])
