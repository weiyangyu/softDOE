"""Compare every mathematical public API with an explicitly versioned installed R MOFAT.

Run: python -m pytest tests/test_mofat_live.py --run-r --mofat-version=2.0
Ordinary tests need neither R nor rpy2. A requested live run fails (never skips)
when the baseline version or R prerequisites are missing.
"""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import softDOE
import softDOE._soft as implementation


@pytest.fixture(scope="module")
def reference(request, tmp_path_factory):
    if not request.config.getoption("--run-r"):
        pytest.skip("live R comparison requires --run-r")
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.fail("--run-r requires Rscript on PATH")
    output = tmp_path_factory.mktemp("mofat") / "reference.json"
    run = subprocess.run(
        [rscript, "--vanilla", str(Path(__file__).with_name("reference_mofat.R")),
         str(output), request.config.getoption("--mofat-version")],
        capture_output=True, text=True, timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(output.read_text())
    assert result["provenance"]["MOFAT"] == request.config.getoption("--mofat-version")
    return result


@pytest.mark.parametrize("case_name", [f"{p}_{l}_{s}" for p,l in [(2,4),(3,2),(4,2),(5,4)]
                                       for s in ("standard","strict")])
def test_live_generators_and_measure(reference, monkeypatch, case_name):
    case = reference["cases"][case_name]
    base = case["base"]
    monkeypatch.setattr(implementation, "_a_fun", lambda p,l,seed: {
        "A": np.asarray(base["A"],dtype=float), "L":base["L"],
        "mod":base["mod"], "quotient":base["quotient"],
    })
    args = dict(p=case["p"],l=case["l"],structure=case["structure"],m=32,backend="cpu")
    result = softDOE.soft(**args)
    np.testing.assert_allclose(result["design"],case["soft"]["design"],rtol=0,atol=2e-14)
    for key in ("Q","normalized_Q"):
        np.testing.assert_allclose(result[key],case["soft"][key],rtol=2e-12,atol=0)
    assert result["qmc_size"] == case["soft"]["qmc_size"]
    assert result["backend_used"] == case["soft"]["backend_used"] == "cpu"
    augmented = softDOE.a_soft(**args,n_aug=3,N=64)
    np.testing.assert_allclose(augmented["design"],case["augmented"]["design"],rtol=0,atol=2e-14)
    for key in ("Q","normalized_Q"):
        np.testing.assert_allclose(augmented[key],case["augmented_Q"][key],rtol=2e-12,atol=0)
    measured = softDOE.measure(case["soft"]["design"],case["y"])
    for key in ("mustar","t"):
        np.testing.assert_allclose(measured[key],case["measure"][key],rtol=2e-12,atol=2e-12)


def test_live_qmc(reference):
    for case in reference["qmc"]:
        actual = softDOE.qmc_generate(case["n"],case["p"],case["threshold"])
        np.testing.assert_array_equal(actual,np.asarray(case["points"],dtype=np.float32).astype(float))


def test_live_Q(reference):
    for case in reference["Q"]:
        actual = softDOE.Q(case["alpha"],case["beta"],case["design"],case["qmc"],backend="cpu")
        for key in ("Q","normalized_Q"):
            np.testing.assert_allclose(actual[key],case["result"][key],rtol=2e-12,atol=0)
        assert actual["backend_used"] == case["result"]["backend_used"] == "cpu"


def test_live_supplied_augmentation(reference):
    case = reference["supplied"]
    actual = softDOE.a_soft(n_aug=3,N=64,ofat_design=case["input"],backend="cpu")
    np.testing.assert_allclose(actual["design"],case["result"]["design"],rtol=0,atol=2e-14)
    np.testing.assert_array_equal(actual["design"][:len(case["input"])],case["input"])
    for key in ("Q", "normalized_Q"):
        np.testing.assert_allclose(actual[key], case["Q"][key], rtol=2e-12, atol=0)


def test_live_mofat_screening(reference):
    case = reference["mofat_measure"]
    actual = softDOE.measure(case["design"],case["y"])
    for key in ("mustar","t"):
        np.testing.assert_allclose(actual[key],case["result"][key],rtol=2e-12,atol=2e-12)
