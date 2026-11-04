"""Final-design quality regressions for CPU search decisions near ties."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

import softDOE
import softDOE._soft as implementation
from softDOE import _native


REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/mofat_cpu_near_ties.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["id"])
@pytest.mark.skipif(sys.platform != "darwin", reason="MOFAT 2.0/macOS Accelerate trajectory reference")
def test_cpu_augmented_quality_matches_mofat_near_ties(case, monkeypatch):
    base = {**case["base"], "A": np.asarray(case["base"]["A"], dtype=float)}
    monkeypatch.setattr(implementation, "_a_fun", lambda p, l, seed: base)
    result = softDOE.a_soft(
        **{key: case[key] for key in ("p", "l", "structure", "m", "N", "n_aug", "seed")},
        backend="cpu",
    )
    assert result["backend_used"] == "cpu"
    # This fixture records macOS Accelerate trajectories. Other BLAS libraries
    # can round exact ties differently; portable Q/solve checks remain below.
    # Match the broad experiment gate: equivalent quality is required, but
    # interchangeable optimal designs need not have identical coordinates.
    for key in ("Q", "normalized_Q"):
        np.testing.assert_allclose(result[key], case["expected"][key], rtol=1e-8, atol=0)


@pytest.mark.parametrize("p", [1, 142, 143, 160])
def test_cpu_scaled_rhs_preserves_one_point_scores_across_underflow(p):
    # For one design row, Q(z) = K(x,z)^2 / (1+nugget). The distant
    # correlation is normal at p=142, subnormal at p=143, and zero at p=160.
    # One evaluation column stays at x, exercising mixed scaling paths.
    design = np.zeros((1, p))
    evaluation = np.vstack((np.zeros(p), np.ones(p)))
    expected = np.array([-math.log1p(1e-6), -2*p*math.log(145)-math.log1p(1e-6)])
    actual = _native.evaluate_cpu(design, evaluation, 1/144)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)
    result = softDOE.Q(1, 1/144, design, evaluation, backend="cpu")
    assert result["normalized_Q"] == pytest.approx(math.exp(expected[1]/p), rel=2e-13)


@pytest.mark.parametrize("n", [63, 64, 65])
def test_cpu_cholesky_panel_boundary_matches_independent_solve(n):
    rng = np.random.default_rng(682)
    design, evaluation = rng.random((n, 3)), rng.random((7, 3))
    beta = 1/64

    def kernel(left, right):
        return np.prod(1/(1+(left[:, None]-right[None, :])**2/beta), axis=2)

    corr = kernel(design, design) + 1e-6*np.eye(n)
    rhs = kernel(design, evaluation)
    expected = np.sum(rhs*np.linalg.solve(corr, rhs), axis=0)
    actual = _native.evaluate_cpu(design, evaluation, beta)
    np.testing.assert_allclose(np.exp(actual), expected, rtol=2e-12, atol=0)
