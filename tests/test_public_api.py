import builtins
import numpy as np
import pytest
import softDOE
from softDOE._engine import ArrayOps, QState


def test_original_reported_call_uses_pydoe(monkeypatch):
    # Emulate a modern install with only lowercase pydoe importable.
    try:
        import pydoe as installed_pydoe
    except ModuleNotFoundError:
        import pyDOE as installed_pydoe
    original_import = builtins.__import__
    calls = []
    def guarded(name, *args, **kwargs):
        if name == "pydoe":
            calls.append(name)
            return installed_pydoe
        if name == "pyDOE":
            raise AssertionError("must prefer the installed lowercase module")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = softDOE.soft(p=10, l=4, seed=42)
    assert result["design"].shape == (44, 10)
    assert result["Q"] > 0
    assert calls == ["pydoe"]


@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_contract_and_block_structure(structure):
    p, l = 4, 4
    result = softDOE.soft(p, l, structure, m=16, seed=4, backend="cpu")
    assert list(result) == ["design", "Q", "normalized_Q", "qmc_size",
                            "backend_used", "device", "fallback_reason"]
    assert result["normalized_Q"] == pytest.approx(result["Q"]**(1/p))
    assert result["qmc_size"] == 32
    blocks = np.split(result["design"], p+1)
    factors = []
    for i in range(p):
        a = blocks[0] if structure == "standard" else blocks[i]
        changed = np.flatnonzero(np.any(~np.isclose(a, blocks[i+1], rtol=0, atol=1e-14),axis=0))
        assert changed.size == 1
        factors.append(changed[0])
    assert sorted(factors) == list(range(p))


def test_augmentation_range_scales_and_supplied_overrides():
    result = softDOE.a_soft(3,2,n_aug=2,m=16,N=64,backend="cpu")
    assert result["design"].shape == (10,3)
    boundary = 0.25/6
    np.testing.assert_allclose([result["design"].min(),result["design"].max()],
                               [boundary,1-boundary])
    existing = np.full((6,2),0.5)
    baseline = existing.copy()
    result = softDOE.a_soft(p=20,l=3,structure=object(),m=object(),seed=object(),
                           n_aug=2,N=16,ofat_design=existing,backend="cpu")
    np.testing.assert_array_equal(existing,baseline)
    np.testing.assert_array_equal(result["design"][:6],existing)
    assert result["design"].shape == (8,2)


def test_invalid_structure_matches_mofat_standard_default():
    for value in ("invalid", None, [], np.array([1,2])):
        expected = softDOE.soft(3,2,m=16,backend="cpu")
        actual = softDOE.soft(3,2,structure=value,m=16,backend="cpu")
        np.testing.assert_array_equal(actual["design"],expected["design"])


@pytest.mark.parametrize("kwargs", [dict(p=True,l=4),dict(p=2,l=3),dict(p=2.5,l=4),
 dict(p=np.nan,l=4),dict(p=2,l=4,m=0),dict(p=2,l=np.inf),dict(p=1025,l=2)])
def test_invalid_soft_parameters(kwargs):
    with pytest.raises((ValueError,TypeError)):
        softDOE.soft(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(n_aug=0),dict(n_aug=9,N=8),dict(n_aug=1,N=False),
 dict(n_aug=1,ofat_design=np.ones((5,2))),dict(n_aug=1,ofat_design=np.ones((6,2))*2),
 dict(n_aug=1,ofat_design=np.full((6,2),np.nan)),dict(n_aug=1,ofat_design=[["x"]])])
def test_invalid_augmentation_parameters(kwargs):
    with pytest.raises((ValueError,TypeError)):
        softDOE.a_soft(p=2,l=2,**kwargs)


@pytest.mark.parametrize("args", [(0,2), (4,0), (True,2), (4,2,-1), (2,1025), (2,30,30)])
def test_invalid_qmc_parameters(args):
    with pytest.raises((ValueError,TypeError)):
        softDOE.qmc_generate(*args)


def test_qmc_order_and_rng_independence():
    before = np.random.get_state()
    first = softDOE.qmc_generate(7,3)
    np.testing.assert_array_equal(first[-8:],[[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                                             [0,0,1],[1,0,1],[0,1,1],[1,1,1]])
    np.testing.assert_array_equal(first,softDOE.qmc_generate(7,3))
    after = np.random.get_state()
    for a,b in zip(before,after):
        np.testing.assert_array_equal(a,b)


def test_log_space_survives_underflow():
    state = QState(np.zeros((2,160)),np.ones((3,160)),1/64,ArrayOps())
    assert np.isfinite(state.score)
    assert np.exp(state.score) == 0
    assert np.exp(state.score/160) > 0


def test_incremental_score_matches_full_factorization():
    rng = np.random.default_rng(23)
    design, points, added = rng.random((8,3)),rng.random((64,3)),rng.random((2,3))
    incremental = QState(design,points,1/64,ArrayOps()).append(added)
    full = QState(np.vstack((design,added)),points,1/64,ArrayOps())
    np.testing.assert_allclose(incremental.log_values,full.log_values,rtol=1e-12,atol=1e-12)
