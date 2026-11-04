"""Numerical regression against independently generated MOFAT 2.0 fixtures."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import softDOE
import softDOE._soft as implementation

REFERENCE = json.loads((Path(__file__).parent / "fixtures/mofat_reference.json").read_text())
Q_REFERENCE = json.loads((Path(__file__).parent / "fixtures/mofat_q.json").read_text())


def use_shared_base(monkeypatch, A):
    def base(p, l, seed):
        factorial = math.factorial(l)
        return dict(A=np.array(A, dtype=float), L=factorial,
                    mod=p % factorial if p >= factorial else None,
                    quotient=p // factorial if p >= factorial else None)
    monkeypatch.setattr(implementation, "_a_fun", base)


@pytest.mark.parametrize("case", REFERENCE["qmc"], ids=lambda c: f"n{c['n']}-p{c['p']}-t{c['threshold']}")
def test_qmc_matches_mofat(case):
    actual = softDOE.qmc_generate(case["n"], case["p"], case["threshold"])
    # JSON has ~16 decimal digit precision, whereas spacefillr values are float32.
    expected = np.asarray(case["points"], dtype=np.float32).astype(float)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name,case", REFERENCE["cases"].items())
def test_soft_matches_mofat_shared_base(name, case, monkeypatch):
    use_shared_base(monkeypatch, case["A"])
    result = softDOE.soft(case["p"], case["l"], case["structure"], m=case["m"], backend="cpu")
    expected = case["soft"]
    assert list(result) == list(expected)
    np.testing.assert_allclose(result["design"], expected["design"], rtol=0, atol=2e-14)
    np.testing.assert_allclose([result["Q"], result["normalized_Q"]],
                               [expected["Q"], expected["normalized_Q"]], rtol=2e-12, atol=0)
    assert result["qmc_size"] == expected["qmc_size"]
    assert result["backend_used"] == "cpu"
    assert result["fallback_reason"] is None


@pytest.mark.parametrize("name,case", REFERENCE["cases"].items())
def test_a_soft_matches_mofat_shared_base(name, case, monkeypatch):
    use_shared_base(monkeypatch, case["A"])
    actual = softDOE.a_soft(case["p"], case["l"], case["structure"], n_aug=3,
                           m=64, N=128, backend="cpu")
    np.testing.assert_allclose(actual["design"], case["a_soft"]["design"], rtol=0, atol=2e-14)
    assert [key for key in actual if key not in ("Q", "normalized_Q")] == list(case["a_soft"])
    expected = Q_REFERENCE["augmented"][name]
    np.testing.assert_allclose([actual["Q"], actual["normalized_Q"]],
                              [expected["Q"], expected["normalized_Q"]], rtol=2e-12, atol=0)


def test_mofat_documentation_examples(monkeypatch):
    case = REFERENCE["examples"]
    use_shared_base(monkeypatch, case["A"])
    soft = softDOE.soft(p=2, l=4, backend="cpu")
    augmented = softDOE.a_soft(p=2, l=4, n_aug=8, backend="cpu")
    np.testing.assert_allclose(soft["design"], case["soft"]["design"], rtol=0, atol=2e-14)
    np.testing.assert_allclose(augmented["design"], case["a_soft"]["design"], rtol=0, atol=2e-14)


def test_supplied_design_augmentation_matches_mofat():
    case = REFERENCE["supplied"]
    result = softDOE.a_soft(n_aug=8, ofat_design=case["input"], backend="cpu")
    np.testing.assert_array_equal(result["design"][:6], case["input"])
    np.testing.assert_allclose(result["design"], case["result"]["design"], rtol=0, atol=2e-14)


@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_friedman_measure_matches_mofat(structure):
    case = REFERENCE["measures"][structure]
    actual = softDOE.measure(case["design"], case["y"])
    for key in ("mustar", "t"):
        np.testing.assert_allclose(actual[key], case["result"][key], rtol=2e-12, atol=2e-12)
