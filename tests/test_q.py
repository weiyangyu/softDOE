"""Public Q and final augmentation scores against independent references."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import softDOE
from softDOE import _backend, _q, _augment
from softDOE._engine import ArrayOps
from softDOE._qmc import _sobol_points

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES / "mofat_q.json").read_text())


@pytest.mark.parametrize("name,case", REFERENCE["cases"].items())
@pytest.mark.parametrize("backend", ["cpu", "metal", "cuda"])
def test_q_matches_mofat(name, case, backend, monkeypatch):
    if backend != "cpu" and not softDOE.backend_info()[backend]["available"]:
        pytest.skip(f"No {backend} device")
    if backend in ("cpu", "metal"):
        monkeypatch.setitem(sys.modules, "torch", None)
    result = softDOE.Q(case["alpha"], case["beta"], case["design"], case["qmc"], backend)
    assert list(result) == ["Q", "normalized_Q", "backend_used"]
    assert result["backend_used"] == backend
    np.testing.assert_allclose([result["Q"], result["normalized_Q"]],
                              [case["result"]["Q"], case["result"]["normalized_Q"]],
                              rtol=5e-4 if backend == "metal" else 2e-12, atol=0)


@pytest.mark.parametrize("alpha", [.25, 1, 2.5])
def test_q_matches_direct_kernel_solve(alpha):
    rng = np.random.default_rng(451)
    x, q = rng.uniform(-2, 2, (9, 4)), rng.uniform(-3, 3, (35, 4))
    beta = .8
    def kernel(a, b):
        return np.prod((1 + (a[:, None, :] - b[None, :, :])**2 / beta)**(-alpha), axis=2)
    rhs = kernel(x, q)
    expected = np.min(np.sum(rhs * np.linalg.solve(kernel(x, x) + 1e-6*np.eye(len(x)), rhs), axis=0))
    result = softDOE.Q(alpha, beta, x, q, "cpu")
    np.testing.assert_allclose(result["Q"], expected, rtol=2e-13)
    np.testing.assert_allclose(result["normalized_Q"], expected**.25, rtol=2e-13)


def test_one_point_and_underflow_normalization():
    assert softDOE.Q(1, .1, [[0]], [[0]], "cpu")["Q"] == pytest.approx(1/(1+1e-6))
    result = softDOE.Q(2.5, 1/144, np.zeros((1,160)), np.ones((1,160)), "cpu")
    assert result["Q"] == 0
    expected = math.exp(-5*math.log(145) - math.log1p(1e-6)/160)
    assert result["normalized_Q"] == pytest.approx(expected, rel=2e-13)


@pytest.mark.parametrize("name", ["alpha", "beta"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True, 1j, "1", [1], None])
def test_invalid_kernel_parameter(name, value):
    args = dict(alpha=1, beta=.1, design=[[0,0]], qmc=[[1,1]], backend="cpu")
    args[name] = value
    with pytest.raises((ValueError, TypeError), match=name):
        softDOE.Q(**args)


@pytest.mark.parametrize("name", ["design", "qmc"])
@pytest.mark.parametrize("value", [[], [1,2], np.empty((0,2)), np.empty((2,0)),
                                   [[np.nan,0]], [[np.inf,0]], [[1j,0]], [["1","0"]], [[True,False]]])
def test_invalid_points(name, value):
    args = dict(alpha=1, beta=.1, design=[[0,0]], qmc=[[1,1]], backend="cpu")
    args[name] = value
    with pytest.raises(ValueError, match=name):
        softDOE.Q(**args)


def test_column_mismatch():
    with pytest.raises(ValueError, match="columns"):
        softDOE.Q(1, .1, [[0,0]], [[0]], "cpu")


@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_q_reproduces_soft_result(structure):
    result = softDOE.soft(4,4,structure,m=64,seed=123,backend="cpu")
    scored = softDOE.Q(1, 1/64, result["design"], softDOE.qmc_generate(64,4), "cpu")
    assert scored["Q"] == pytest.approx(result["Q"], rel=2e-12)
    assert scored["normalized_Q"] == pytest.approx(result["normalized_Q"], rel=2e-12)


@pytest.mark.parametrize("structure", ["standard", "strict"])
@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("n_aug,N", [(1,32), (8,64), (4,4)])
@pytest.mark.parametrize("backend", ["cpu", "metal", "cuda"])
def test_augmented_scores_evaluate_complete_final_design(structure, supplied, n_aug, N, backend):
    if backend != "cpu" and not softDOE.backend_info()[backend]["available"]:
        pytest.skip(f"No {backend} device")
    base = np.linspace(.1,.9,12).reshape(6,2) if supplied else None
    result = softDOE.a_soft(2,2,structure,n_aug=n_aug,m=16,N=N,
                            ofat_design=base,seed=123,backend=backend)
    assert list(result) == ["design", "Q", "normalized_Q", "backend_used", "device", "fallback_reason"]
    assert result["backend_used"] == backend, result["fallback_reason"]
    b = .25/(4+n_aug)
    pool = b + _sobol_points(N,2)*(1-2*b)
    full = softDOE.Q(1, (4+n_aug)**-2, result["design"], pool, "cpu")
    tolerance = 5e-4 if backend == "metal" else 2e-12
    assert result["Q"] == pytest.approx(full["Q"], rel=tolerance)
    assert result["normalized_Q"] == pytest.approx(full["normalized_Q"], rel=tolerance)
    before_last = softDOE.Q(1, (4+n_aug)**-2, result["design"][:-1], pool, "cpu")
    assert full["Q"] > before_last["Q"]
    if supplied:
        np.testing.assert_array_equal(result["design"][:6], base)


def test_q_cpu_does_not_probe_gpu(monkeypatch):
    def forbidden():
        raise AssertionError("CPU must not probe GPU")
    monkeypatch.setattr(_backend, "backend_info", forbidden)
    assert softDOE.Q(1, .1, [[0]], [[1]], "cpu")["backend_used"] == "cpu"


@pytest.mark.parametrize("backend", ["bad", None, [], "metal", "cuda"])
def test_q_unavailable_backend_falls_back(monkeypatch, backend):
    monkeypatch.setattr(_backend, "backend_info", lambda: {
        key: dict(available=False, device=None, reason="Unavailable") for key in ("metal", "cuda")})
    assert softDOE.Q(1, .1, [[0]], [[1]], backend) == softDOE.Q(1, .1, [[0]], [[1]], "cpu")


def test_q_runtime_failure_restarts_on_cpu(monkeypatch):
    original = _q._score
    calls = []
    def failing(alpha, beta, design, qmc, ops):
        calls.append(ops.backend)
        if ops.backend == "metal":
            raise RuntimeError("simulated Q failure")
        return original(alpha, beta, design, qmc, ops)
    monkeypatch.setattr(_backend, "_resolve_backend", lambda b: ("metal", "GPU", None))
    monkeypatch.setattr(_q, "_score", failing)
    assert softDOE.Q(1, .1, [[0]], [[1]])["backend_used"] == "cpu"
    assert calls == ["metal", "cpu"]


@pytest.mark.parametrize("supplied", [False, True])
def test_final_augmentation_score_failure_restarts_workflow(monkeypatch, supplied):
    base = np.full((6,2), .5) if supplied else None
    args = dict(p=2,l=2,n_aug=2,m=16,N=32,seed=123,ofat_design=base)
    expected = softDOE.a_soft(**args,backend="cpu")
    original_optimize, original_augment = _augment._optimize, _augment._augment
    calls = []
    def optimize(prepared, structure, ops):
        calls.append("base-"+ops.backend)
        return original_optimize(prepared, structure, ArrayOps())
    def augment(design, candidates, n_aug, beta, ops, return_score=False):
        calls.append("augment-"+ops.backend)
        result = original_augment(design, candidates, n_aug, beta, ArrayOps(), return_score)
        if ops.backend == "metal" and return_score:
            raise RuntimeError("simulated final-score failure")
        return result
    monkeypatch.setattr(_augment, "_optimize", optimize)
    monkeypatch.setattr(_augment, "_augment", augment)
    monkeypatch.setattr(_backend, "_resolve_backend", lambda b: ("metal", "GPU", None))
    actual = softDOE.a_soft(**args,backend="metal")
    assert calls == (["augment-metal","augment-cpu"] if supplied else
                      ["base-metal","augment-metal","base-cpu","augment-cpu"])
    assert actual["backend_used"] == "cpu"
    assert "final-score failure" in actual["fallback_reason"]
    np.testing.assert_array_equal(actual["design"], expected["design"])
    assert actual["Q"] == expected["Q"]


@pytest.mark.parametrize("alpha", [.25,1,2.5])
def test_cuda_general_alpha_and_final_augmentation_on_host(alpha):
    torch = pytest.importorskip("torch")
    ops = ArrayOps()
    ops.backend, ops.torch, ops.device, ops.dtype = "cuda", torch, "cpu", torch.float64
    rng = np.random.default_rng(53)
    x, q = rng.random((6,3)), rng.random((64,3))
    actual = _q._score(alpha, 1/64, x, q, ops)
    expected = softDOE.Q(alpha, 1/64, x, q, "cpu")["Q"]
    assert math.exp(actual) == pytest.approx(expected, rel=2e-12)
    design, score = _augment._augment(x,q,3,1/121,ops,return_score=True)
    assert math.exp(score) == pytest.approx(softDOE.Q(1,1/121,design,q,"cpu")["Q"],rel=2e-12)
