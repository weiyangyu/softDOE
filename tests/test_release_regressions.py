"""Regression cases found while exercising release-facing numerical inputs."""
import builtins
from types import SimpleNamespace

import numpy as np
import pytest

import softDOE
from softDOE import _backend
from softDOE._engine import ArrayOps, QState
from softDOE._soft import _maximin_lhd_levels


@pytest.mark.parametrize("scale", [1e-200, 1e200, 1e308])
def test_measure_responses_keep_total_effects_under_rescaling(scale):
    design = np.array([[.25, .25], [.75, .75], [.75, .25], [.25, .75],
                       [.25, .75], [.75, .25]])
    y = design @ np.array([1., -1.])
    expected = softDOE.measure(design, y)
    with np.errstate(all="raise"):
        actual = softDOE.measure(design, y * scale)
    np.testing.assert_allclose(actual["mustar"] / scale, expected["mustar"], rtol=1e-14)
    np.testing.assert_allclose(actual["t"], expected["t"], rtol=1e-14)


def test_measure_large_finite_steps_keep_linear_elementary_effects():
    design = np.array([[-1e308], [1e308]])
    with np.errstate(all="raise"):
        actual = softDOE.measure(design, design[:, 0])
    np.testing.assert_allclose(actual["mustar"], [1.])
    np.testing.assert_allclose(actual["t"], [1.])


def test_measure_constant_extreme_responses_have_undefined_total_effects():
    design = np.array([[0.], [1.]])
    with np.errstate(all="raise"):
        actual = softDOE.measure(design, [1e308, 1e308])
    np.testing.assert_array_equal(actual["mustar"], [0.])
    assert np.isnan(actual["t"]).all()


def test_measure_small_contrast_survives_unrelated_large_responses():
    design = np.array([[.1], [.2], [.3], [.4]])
    result = softDOE.measure(design, [1e300, 1e-300, 1e300, 2e-300])
    assert result["mustar"][0] == pytest.approx(2.5e-300, rel=1e-14, abs=0)


@pytest.mark.parametrize("beta, design, points", [(.1, [[0.]], [[1.]]),
                                                (1., [[0., 0.]], [[1., 1.]])])
def test_q_exponents_beyond_log_range_return_zero(beta, design, points):
    result = softDOE.Q(1e308, beta, design, points, backend="cpu")
    assert result["Q"] == result["normalized_Q"] == 0.
    state = QState(design, points, beta, ArrayOps(), alpha=1e308)
    assert state.score == -np.inf
    assert state.append(design).score == -np.inf


def test_gpu_intermediate_overflow_falls_back_to_stable_cpu(monkeypatch):
    class OverflowingGpuOps(ArrayOps):
        def __init__(self):
            super().__init__()
            self.backend = "cuda"

        def correlation(self, x, y, beta, log=False, alpha=1.):
            if log:
                # Emulate a GPU square overflowing before division by beta.
                return np.full((len(x), len(y)), -np.inf)
            return super().correlation(x, y, beta, log=log, alpha=alpha)

    monkeypatch.setattr(_backend, "ArrayOps", lambda backend="cpu":
                        OverflowingGpuOps() if backend == "cuda" else ArrayOps())
    monkeypatch.setattr(_backend, "_resolve_backend", lambda value: ("cuda", "GPU", None))
    result = softDOE.Q(1., 1e308, [[0.]], [[1e200]])
    assert result["backend_used"] == "cpu"
    assert result["Q"] == pytest.approx(1e-184 / (1 + 1e-6), rel=1e-12, abs=0.)


@pytest.mark.parametrize("alpha", [.25, 1., 2.5])
@pytest.mark.parametrize("beta, x, z", [(1e-320, 0., 0.), (1e-320, 0., 1e-160),
                                       (1., -1e160, 1e160), (1., -1e308, 1e308)])
def test_q_extreme_finite_inputs_match_one_point_formula(alpha, beta, x, z):
    # The one-point criterion has a closed form, independent of factorization.
    if x == z:
        term = 0.
    else:
        log_distance = np.log(abs(x / 2 - z / 2)) + np.log(2.)
        term = np.logaddexp(0., 2 * log_distance - np.log(beta))
    expected_log = -2 * alpha * term - np.log1p(1e-6)
    actual = softDOE.Q(alpha, beta, [[x]], [[z]], backend="cpu")
    assert actual["Q"] == pytest.approx(np.exp(expected_log), rel=2e-12, abs=0.)
    assert actual["normalized_Q"] == actual["Q"]
    state = QState([[x]], [[z]], beta, ArrayOps(), alpha=alpha)
    assert state.score == pytest.approx(expected_log, rel=2e-12, abs=1e-14)


def test_legacy_pydoe_unseeded_calls_draw_new_entropy_without_global_rng_changes(monkeypatch):
    original_import = builtins.__import__
    seeds = iter([1, 2])

    def lhs(n, samples, criterion, iterations):
        return np.random.random((samples, n))

    def importing(name, *args, **kwargs):
        if name == "pydoe":
            return SimpleNamespace(lhs=lhs)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    monkeypatch.setattr(np.random, "SeedSequence", lambda: SimpleNamespace(
        generate_state=lambda n: np.array([next(seeds)], dtype=np.uint32)))
    before = np.random.get_state()
    first = _maximin_lhd_levels(4, 2)
    second = _maximin_lhd_levels(4, 2)
    assert not np.array_equal(first, second)
    for old, new in zip(before, np.random.get_state()):
        np.testing.assert_array_equal(old, new)
