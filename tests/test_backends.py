import numpy as np
import pytest
import softDOE
from softDOE import _backend, _augment
from softDOE._engine import ArrayOps, QState


def test_capability_contract_and_refresh():
    info = softDOE.backend_info(refresh=True)
    assert list(info) == ["cpu", "metal", "cuda"]
    for entry in info.values():
        assert set(entry) == {"compiled", "available", "device", "reason"}
        assert type(entry["compiled"]) is bool
        assert type(entry["available"]) is bool
    assert info["cpu"] == dict(compiled=True,available=True,device="CPU",reason=None)
    info["cpu"]["device"] = "mutated"
    assert softDOE.backend_info()["cpu"]["device"] == "CPU"
    with pytest.raises(TypeError):
        softDOE.backend_info(refresh=1)


@pytest.mark.parametrize("backend", ["bad", [], None, "CUDA", "metal"])
def test_invalid_or_unavailable_backend_falls_back(monkeypatch, backend):
    monkeypatch.setattr(_backend, "backend_info", lambda: {
        "cpu":dict(available=True,device="CPU",reason=None),
        "cuda":dict(available=False,device=None,reason="CUDA unavailable"),
        "metal":dict(available=False,device=None,reason="Metal unavailable")})
    expected = softDOE.soft(3,2,m=16,backend="cpu")
    actual = softDOE.soft(3,2,m=16,backend=backend)
    assert actual["backend_used"] == "cpu"
    assert actual["device"] == "CPU"
    assert actual["fallback_reason"]
    np.testing.assert_array_equal(actual["design"],expected["design"])


def test_explicit_cpu_does_not_probe_gpu(monkeypatch):
    def forbidden():
        raise AssertionError("GPU runtime must not be imported for CPU requests")
    monkeypatch.setattr(_backend, "backend_info", forbidden)
    assert softDOE.soft(3,2,m=8,backend="cpu")["backend_used"] == "cpu"


@pytest.mark.parametrize("system,expected", [("Darwin","metal"),("Linux","cuda"),("Windows","cpu")])
def test_auto_platform_policy(monkeypatch, system, expected):
    monkeypatch.setattr(_backend.platform,"system",lambda:system)
    # Follow the platform restrictions enforced by real capability discovery.
    monkeypatch.setattr(_backend,"backend_info",lambda:{
        "metal":dict(available=system=="Darwin",device="Metal",reason=None),
        "cuda":dict(available=system=="Linux",device="CUDA",reason=None)})
    assert _backend._resolve_backend("auto")[0] == expected


@pytest.mark.parametrize("supplied", [False,True])
def test_gpu_failure_restarts_correct_workflow(monkeypatch,supplied):
    existing = np.full((6,2),0.5) if supplied else None
    args = dict(p=2,l=2,n_aug=2,m=16,N=32,ofat_design=existing,seed=42)
    expected = softDOE.a_soft(**args,backend="cpu")
    calls=[]
    original_optimize, original_augment = _augment._optimize,_augment._augment
    class SimulatedOps(ArrayOps):
        def __init__(self,backend="cpu"):
            super().__init__()
            self.backend=backend
    monkeypatch.setattr(_backend,"ArrayOps",SimulatedOps)
    monkeypatch.setattr(_backend,"_resolve_backend",lambda b:("metal","Test GPU",None))
    def optimize(prepared,structure,ops):
        calls.append("opt-"+ops.backend)
        return original_optimize(prepared,structure,ArrayOps())
    def augment(design,candidates,n_aug,beta,ops,return_score=False):
        calls.append("augment-"+ops.backend)
        if ops.backend=="metal":
            raise RuntimeError("simulated GPU failure")
        return original_augment(design,candidates,n_aug,beta,ops,return_score)
    monkeypatch.setattr(_augment,"_optimize",optimize)
    monkeypatch.setattr(_augment,"_augment",augment)
    actual=softDOE.a_soft(**args,backend="metal")
    assert calls == (["augment-metal","augment-cpu"] if supplied else
                      ["opt-metal","augment-metal","opt-cpu","augment-cpu"])
    assert actual["backend_used"] == "cpu"
    assert actual["device"] == "CPU"
    assert "augmentation" in actual["fallback_reason"]
    np.testing.assert_array_equal(actual["design"],expected["design"])


@pytest.mark.parametrize("backend",["cuda","metal"])
def test_actual_gpu_when_available(backend):
    if not softDOE.backend_info()[backend]["available"]:
        pytest.skip(softDOE.backend_info()[backend]["reason"])
    gpu=softDOE.soft(3,2,m=32,backend=backend)
    cpu=softDOE.soft(3,2,m=32,backend="cpu")
    assert gpu["backend_used"] == backend
    np.testing.assert_allclose(gpu["Q"],cpu["Q"],rtol=5e-4)
    for structure in ("standard","strict"):
        gpu=softDOE.a_soft(3,2,structure,n_aug=2,m=32,N=128,backend=backend)
        cpu=softDOE.a_soft(3,2,structure,n_aug=2,m=32,N=128,backend="cpu")
        assert gpu["backend_used"] == backend
        np.testing.assert_allclose(gpu["design"],cpu["design"],atol=2e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_torch_numerical_path_on_host(dtype):
    torch=pytest.importorskip("torch")
    ops=ArrayOps()
    ops.torch,ops.device,ops.dtype=torch,"cpu",getattr(torch,dtype)
    rng=np.random.default_rng(5)
    design,points,block=rng.random((8,3)),rng.random((64,3)),rng.random((2,3))
    actual=QState(design,points,1/64,ops).append(block)
    expected=QState(np.vstack((design,block)),points,1/64,ArrayOps())
    np.testing.assert_allclose(ops.numpy(actual.log_values),expected.log_values,
                               rtol=1e-5 if dtype=="float32" else 1e-12)
