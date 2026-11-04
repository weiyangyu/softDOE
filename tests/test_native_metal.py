"""Native Metal public paths work independently of optional PyTorch."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import softDOE
from softDOE import _native
from softDOE._metal import shader_source, MetalOps
from softDOE._engine import ArrayOps, QState
from softDOE._soft import _prepare, _optimize
from softDOE._augment import _augment


@pytest.fixture
def native_metal():
    if not hasattr(_native, 'metal_info'):
        pytest.skip('No native Metal build')
    info = _native.metal_info(shader_source())
    if not info['available']:
        pytest.skip(info['reason'])
    assert softDOE.backend_info(refresh=True)['metal']['available']
    return MetalOps()


@pytest.mark.parametrize('structure', ['standard', 'strict'])
def test_native_metal_matches_mofat_without_torch(native_metal, structure, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    r = json.loads((Path(__file__).parent/'fixtures'/f'mofat_{structure}_metal.json').read_text())
    expected = r.get('metal', r)
    prepared = _prepare(r['p'], r['l'], r['m'], 123, 2*r['l'])
    prepared['A'] = np.asarray(r['A'])
    x, score = _optimize(prepared, structure, native_metal)
    np.testing.assert_allclose(x, expected['design'], rtol=0, atol=2e-14)
    np.testing.assert_allclose(score, expected['log_Q'], rtol=0, atol=3e-5)
    cpu = QState(x, prepared['qmc'], prepared['beta'], ArrayOps())
    np.testing.assert_allclose(score, cpu.score, rtol=0, atol=3e-5)
    # Discovery and the full public workflow must not import PyTorch either.
    assert softDOE.backend_info(refresh=True)['metal']['available']
    result = softDOE.a_soft(3,2,structure,n_aug=3,m=16,N=128,backend='metal')
    assert result['backend_used'] == 'metal', result['fallback_reason']


@pytest.mark.parametrize('n,p,N,count', [(6,2,128,8), (44,10,1000,12), (126,20,10000,20)])
def test_native_augmentation_preserves_base_and_cpu_choices(native_metal, n, p, N, count):
    rng = np.random.default_rng(123)
    x, pool = rng.random((n,p)), rng.random((N,p))
    expected = _augment(x,pool,count,1/1024,ArrayOps())
    actual = _augment(x,pool,count,1/1024,native_metal)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:n],x)


def test_native_metal_underflow_remains_finite(native_metal):
    base = np.zeros((2,100))
    points = np.ones((7,100))
    for strict in [False,True]:
        x, q = _native.optimize_metal(base,points,1/144,0.1,0,0,strict,shader_source())
        ref = _native.evaluate_cpu(x,points,1/144).min()
        assert np.isfinite(q) and q < -745
        np.testing.assert_allclose(q,ref,rtol=3e-6)
