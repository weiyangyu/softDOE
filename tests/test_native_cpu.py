"""Independent numerical checks for compiled Q and incremental search."""
import numpy as np
import pytest

from softDOE import _native
from softDOE._engine import ArrayOps, QState
from softDOE._soft import _prepare, _optimize
from softDOE._augment import _augment


@pytest.mark.parametrize("p,n,m,k", [(3, 8, 64, 2), (20, 12, 1000, 6), (160, 2, 8, 1)])
def test_native_q_and_append_agree_with_independent_scipy(p, n, m, k):
    rng = np.random.default_rng(42)
    x, q, extra = rng.random((n, p)), rng.random((m, p)), rng.random((k, p))
    original = QState(x, q, 1/144, ArrayOps())
    for block, expected in [(None, original), (extra, original.append(extra))]:
        actual = _native.evaluate_cpu(x, q, 1/144, block)
        np.testing.assert_allclose(actual, expected.log_values, rtol=2e-12, atol=2e-12)


def test_native_underflow_and_duplicate_rows():
    x, q, extra = np.zeros((4, 160)), np.ones((3, 160)), np.full((2, 160), 0.1)
    actual = _native.evaluate_cpu(x, q, 1/144, extra)
    expected = QState(x, q, 1/144, ArrayOps()).append(extra).log_values
    assert np.isfinite(actual).all()
    assert (np.exp(actual) == 0).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("p,l,m", [(2, 4, 64), (3, 2, 64), (4, 4, 64), (10, 4, 64), (20, 6, 500)])
@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_cached_native_search_matches_reference(p, l, m, structure):
    prepared = _prepare(p, l, m, 123, 2*l)
    reference_ops = ArrayOps()
    reference_ops.native = False
    expected, q = _optimize(prepared, structure, reference_ops)
    actual, score = _optimize(prepared, structure, ArrayOps())
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(score, q, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("n,p,N,count", [(6, 2, 128, 8), (44, 10, 1000, 12), (126, 20, 10000, 20)])
def test_native_augmentation_matches_reference(n, p, N, count):
    rng = np.random.default_rng(123)
    x, pool = rng.random((n, p)), rng.random((N, p))
    base, candidates = x.copy(), pool.copy()
    reference_ops = ArrayOps()
    reference_ops.native = False
    expected = _augment(x, pool, count, 1/1024, reference_ops)
    actual = _augment(x, pool, count, 1/1024, ArrayOps())
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, base)
    np.testing.assert_array_equal(pool, candidates)
