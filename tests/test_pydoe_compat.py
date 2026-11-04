"""The initialization must remain PyDOE's centered maximin LHD."""
import builtins
from types import SimpleNamespace

import numpy as np
import pytest
from softDOE._soft import _maximin_lhd_levels


@pytest.mark.parametrize("spelling",["pydoe","pyDOE"])
def test_pydoe_import_spelling_and_exact_settings(monkeypatch,spelling):
    captured={}
    expected=np.array([[.125,.625],[.375,.125],[.625,.875],[.875,.375]])
    def lhs(n,samples,criterion,iterations,seed):
        captured.update(n=n,samples=samples,criterion=criterion,iterations=iterations,seed=seed)
        return expected
    original=builtins.__import__
    def importing(name,*args,**kwargs):
        if name in ("pydoe","pyDOE"):
            if name==spelling:
                return SimpleNamespace(lhs=lhs)
            raise ModuleNotFoundError(f"No module named '{name}'",name=name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",importing)
    actual=_maximin_lhd_levels(4,2,seed=123)
    assert captured==dict(n=2,samples=4,criterion="centermaximin",iterations=100,seed=123)
    np.testing.assert_array_equal(actual,expected*4+.5)


def test_matches_installed_pydoe_lhs():
    try:
        from pydoe import lhs
    except ModuleNotFoundError:
        from pyDOE import lhs
    expected=lhs(n=10,samples=4,criterion="centermaximin",iterations=100,seed=42)*4+.5
    np.testing.assert_array_equal(_maximin_lhd_levels(4,10,seed=42),expected)


def test_broken_dependency_is_not_hidden_by_legacy_fallback(monkeypatch):
    original=builtins.__import__
    def importing(name,*args,**kwargs):
        if name=="pydoe":
            raise ModuleNotFoundError("missing dependency",name="some_dependency")
        if name=="pyDOE":
            raise AssertionError("must not hide an import failure inside pydoe")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",importing)
    with pytest.raises(ModuleNotFoundError,match="missing dependency"):
        _maximin_lhd_levels(4,2,seed=42)


def test_legacy_global_rng_is_restored(monkeypatch):
    original=builtins.__import__
    def lhs(n,samples,criterion,iterations):
        return np.column_stack([(np.random.permutation(samples)+.5)/samples for _ in range(n)])
    def importing(name,*args,**kwargs):
        if name=="pydoe":
            return SimpleNamespace(lhs=lhs)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",importing)
    before=np.random.get_state()
    first=_maximin_lhd_levels(4,2,seed=42)
    second=_maximin_lhd_levels(4,2,seed=42)
    np.testing.assert_array_equal(first,second)
    for old,new in zip(before,np.random.get_state()):
        np.testing.assert_array_equal(old,new)
