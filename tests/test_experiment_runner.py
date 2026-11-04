"""Integrity checks for experiment evidence, without running R or benchmarks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import run_mofat_experiments as runner
from experiment_cases import experiment_cases


def _write_records(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    # Synthetic consistent records isolate report validation from optimizers.
    case = dict(id="fixture", function="soft", p=2, l=2, structure="standard",
                m=1, seed=1, initialization="shared", tag="minimum",
                base=None, supplied_design=None)
    design = [[.0625, .0625], [.9375, .9375], [.6458333333333334, .0625],
              [.35416666666666663, .9375], [.0625, .6458333333333334],
              [.9375, .35416666666666663]]
    score = dict(Q=.25, normalized_Q=.5, backend_used="cpu")
    actual = dict(id=case["id"], backend="cpu", status="ok", design=design,
                  Q=.25, normalized_Q=.5, cpu_score=score, seconds=[.02, .03],
                  qmc_size=5, backend_used="cpu", device="CPU", fallback_reason=None)
    reference = {**deepcopy(actual), "seconds": [.04, .05], "scored_seconds": [.04, .05]}
    cross = dict(id=case["id"], backend="cpu", status="ok", score=score)
    manifest = dict(profile="smoke", cases=[case], backends=["cpu"],
                    r_backends=["cpu"], r_backend="same", repeats=2)
    return dict(output=tmp_path, manifest=manifest, python=[actual],
                reference=[reference], cross=[cross])


def _summarize(evidence):
    output = evidence["output"]
    _write_records(output / "r-results.jsonl", evidence["reference"])
    _write_records(output / "r-python-scores.jsonl", evidence["cross"])
    return runner.summarize(output, evidence["manifest"], evidence["python"])


def test_complete_consistent_evidence_produces_one_pass(evidence):
    result = _summarize(evidence)
    assert result["status"] == "passed"
    assert result["executions"] == result["passed"] == 1
    assert result["failed"] == 0
    assert result["results"][0]["r_over_python_scored_time"] == pytest.approx(1.8)


@pytest.mark.parametrize("source", ["python", "reference", "cross"])
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unexpected"])
def test_incomplete_or_ambiguous_records_cannot_produce_green_report(evidence, source, corruption):
    records = evidence[source]
    if corruption == "missing":
        records.clear()
    elif corruption == "duplicate":
        records.append(deepcopy(records[0]))
    else:
        records.append({**deepcopy(records[0]), "id": "not_in_manifest"})
    with pytest.raises(ValueError):
        _summarize(evidence)


def test_every_requested_backend_requires_results(evidence):
    evidence["manifest"]["backends"].append("cuda")
    evidence["manifest"]["r_backends"].append("cuda")
    with pytest.raises(ValueError):
        _summarize(evidence)


def test_reference_qmc_size_is_verified(evidence):
    evidence["reference"][0]["qmc_size"] = 999
    result = _summarize(evidence)
    assert result["status"] == "failed"
    assert result["failed"] == 1
    assert result["validation_failed"] == 1
    assert result["validation_passed"] == result["agreement_failed"] == 0
    assert result["results"][0]["validation_status"] == "failed"


@pytest.mark.parametrize("source,key", [("python", "seconds"),
                                        ("reference", "seconds"),
                                        ("reference", "scored_seconds")])
@pytest.mark.parametrize("timings", [[0., .03], [-.01, .03], [float("nan"), .03], [.02]])
def test_invalid_or_incomplete_timing_samples_are_not_reported_as_passes(evidence, source, key, timings):
    evidence[source][0][key] = timings
    result = _summarize(evidence)
    assert result["status"] == "failed"
    assert result["failed"] == 1


@pytest.mark.parametrize("failure", ["score", "shape"])
def test_validation_failure_retains_returned_design_scores_and_times(evidence, monkeypatch, failure):
    import softDOE

    actual = evidence["python"][0]
    returned = {**deepcopy(actual), "design": np.asarray(actual["design"]),
                "Q": .125, "normalized_Q": np.sqrt(.125)}
    if failure == "shape":
        returned["design"] = returned["design"][:-1]
    monkeypatch.setattr(softDOE, "soft", lambda **kwargs: returned)
    monkeypatch.setattr(softDOE, "Q", lambda *args, **kwargs: deepcopy(actual["cpu_score"]))
    monkeypatch.setattr(runner, "evaluation", lambda case: np.zeros((5, 2)))
    record = runner.run_python_case(evidence["manifest"]["cases"][0], "cpu", repeats=2)
    assert record["status"] == "error"
    np.testing.assert_array_equal(record["design"], returned["design"])
    assert record["Q"] == returned["Q"]
    assert record["normalized_Q"] == returned["normalized_Q"]
    assert len(record["seconds"]) == 2
    assert record.get("error")
    if failure == "score":
        assert record["cpu_score"] == actual["cpu_score"]


def test_optimized_interpreter_is_rejected_before_any_experiment():
    result = subprocess.run(
        [sys.executable, "-O", str(Path(runner.__file__).resolve()),
         "--profile", "smoke", "--rscript", "missing-rscript-for-guard-test"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "optimi" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("nonzero", [1e-200, 1e-320, np.nextafter(0., 1.)])
def test_zero_does_not_match_nonzero_q_even_below_normal_float_range(nonzero):
    actual = dict(Q=0., normalized_Q=.01)
    expected = dict(Q=nonzero, normalized_Q=.01)
    with pytest.raises(AssertionError):
        runner.compare_scores(actual, expected, "cpu")


def test_genuine_raw_underflow_can_match_with_nonzero_normalized_score():
    score = dict(Q=0., normalized_Q=.01)
    errors = runner.compare_scores(score, score, "cpu")
    assert errors["Q_relative_error"] == errors["normalized_Q_relative_error"] == 0


@pytest.mark.parametrize("function", ["soft", "a_soft"])
def test_shared_base_transport_uses_integer_ranks_without_changing_python_levels(monkeypatch, function):
    import softDOE._soft as implementation

    case = next(case for case in experiment_cases("smoke")
                if case["function"] == function and case["p"] == 3
                and case["l"] == 4 and case["structure"] == "standard")
    ranks = np.array([[1., 2., 3.], [2., 4., 1.], [3., 1., 4.], [4., 3., 2.]])
    # Centered PyDOE coordinates can reconstruct integer ranks one ULP away.
    raw = np.nextafter(ranks, np.where(ranks == 1, np.inf, 0.))
    saved = raw.copy()
    base = dict(A=raw, L=24, mod=None, quotient=None)
    monkeypatch.setattr(implementation, "_a_fun", lambda p, l, seed: base)

    prepared = runner.prepare_cases([case])[0]
    transported = np.asarray(prepared["base"]["A"])
    np.testing.assert_array_equal(transported, ranks)
    np.testing.assert_array_equal(raw, saved)
    levels = np.linspace(.25 / 8, 1 - .25 / 8, 8)
    original_levels = implementation._level_trans(saved, 4, levels)
    np.testing.assert_array_equal(implementation._level_trans(transported, 4, levels), original_levels)
    # R's numeric vector indices truncate instead of rounding. Canonical ranks
    # must select the same levels under that rule after JSON transport.
    restored = np.asarray(json.loads(json.dumps(transported.tolist())))
    r_indices = np.where(restored < 2.5, 2 * restored - 1, 2 * restored).astype(int) - 1
    np.testing.assert_array_equal(levels[r_indices], original_levels)


def test_different_native_initializers_are_not_mislabeled_as_parity(evidence):
    evidence["manifest"]["cases"][0]["initialization"] = "native"
    evidence["reference"][0].update(Q=.16, normalized_Q=.4,
                                    cpu_score=dict(Q=.16, normalized_Q=.4, backend_used="cpu"))
    result = _summarize(evidence)
    assert result["status"] == "passed"
    row = result["results"][0]
    assert row["comparison"] == "independent_initializers"
    assert row["normalized_quality_relative_difference"] == pytest.approx(.25)


def test_shared_initializer_score_regression_is_a_failure(evidence):
    evidence["reference"][0].update(Q=.16, normalized_Q=.4,
                                    cpu_score=dict(Q=.16, normalized_Q=.4, backend_used="cpu"))
    result = _summarize(evidence)
    assert result["status"] == "failed"
    assert result["failed"] == 1
    assert result["validation_passed"] == result["agreement_failed"] == 1
    assert result["validation_failed"] == 0
    assert result["results"][0]["validation_status"] == "passed"
    assert result["results"][0]["agreement_status"] == "failed"


def test_full_case_matrix_includes_branch_and_pool_boundaries():
    cases = experiment_cases("full")
    assert len({case["id"] for case in cases}) == len(cases)
    valid = [case for case in cases if not case.get("expect_error")]
    assert {case["p"] for case in valid if case["tag"] == "factorial"} >= {23, 24, 25}
    assert {case["initialization"] for case in valid} == {"shared", "native"}
    assert {case.get("supplied") for case in valid} >= {"ofat", "boundary", "duplicate"}
    assert any(case.get("n_aug") == case.get("N") == 1 for case in valid)
    assert {case["structure"] for case in valid} >= {"standard", "strict", "invalid"}


def test_stress_cases_include_dimension_limit_and_underflow_scenarios():
    stress = [case for case in experiment_cases("stress") if case["tag"] == "stress"]
    assert {case["p"] for case in stress} >= {256, 1024}
    assert {case["function"] for case in stress} == {"soft", "a_soft"}
