"""CUDA arithmetic on host tensors, plus actual hardware when available."""
import numpy as np
import pytest

from softDOE._engine import ArrayOps, QState
from softDOE._cuda import PairedUpdates
from softDOE._soft import _prepare, _optimize
from softDOE._augment import _augment


@pytest.fixture(params=['cpu', 'cuda'])
def cuda_ops(request):
    torch = pytest.importorskip('torch')
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('No CUDA device; host arithmetic is tested separately')
    ops = ArrayOps()
    ops.backend, ops.torch, ops.device, ops.dtype = 'cuda', torch, request.param, torch.float64
    return ops


@pytest.mark.parametrize('p,n,m', [(3, 5, 17), (33, 8, 128), (160, 3, 4096), (33, 33, 3000)])
def test_chunked_correlations_match_cpu(cuda_ops, p, n, m):
    rng = np.random.default_rng(42)
    x, q = rng.random((n,p)), rng.random((m,p))
    actual = cuda_ops.correlation(cuda_ops.array(x), cuda_ops.array(q), 1/144, log=True)
    expected = ArrayOps().correlation(x, q, 1/144, log=True)
    np.testing.assert_allclose(cuda_ops.numpy(actual), expected, rtol=2e-13, atol=2e-12)


@pytest.mark.parametrize('duplicates', [False, True])
def test_paired_updates_preserve_both_scores_and_accepted_state(cuda_ops, duplicates):
    rng = np.random.default_rng(13)
    x, q = rng.random((6,5)), rng.random((129,5))
    if duplicates:
        x[1:] = x[0]
    state = QState(x, q, 1/64, cuda_ops)
    for choice in [0, 1, 1, 0]:
        blocks = [rng.random((2,5)), rng.random((2,5))]
        updates = PairedUpdates(state, *blocks)
        expected = [QState(np.vstack((x, b)), q, 1/64, ArrayOps()) for b in blocks]
        np.testing.assert_allclose(updates.scores, [v.score for v in expected], rtol=2e-12, atol=2e-12)
        state = updates.accept(choice)
        np.testing.assert_allclose(cuda_ops.numpy(state.log_values), expected[choice].log_values,
                                   rtol=2e-12, atol=2e-12)
        x = np.vstack((x, blocks[choice]))


@pytest.mark.parametrize('structure', ['standard', 'strict'])
def test_search_and_augmentation_agree_with_unbatched_state(cuda_ops, structure):
    prepared = _prepare(4,4,64,123,8)
    expected_ops = ArrayOps()
    expected_ops.backend, expected_ops.torch = cuda_ops.backend, cuda_ops.torch
    expected_ops.device, expected_ops.dtype, expected_ops.native = cuda_ops.device, cuda_ops.dtype, False
    x, score = _optimize(prepared, structure, cuda_ops)
    ref, q = _optimize(prepared, structure, expected_ops)
    np.testing.assert_array_equal(x, ref)
    np.testing.assert_allclose(score, q, rtol=2e-12, atol=2e-12)
    points = np.random.default_rng(3).random((128,4))
    actual = _augment(x, points, 4, 1/144, cuda_ops)
    ref = _augment(x, points, 4, 1/144, ArrayOps())
    np.testing.assert_array_equal(actual, ref)
