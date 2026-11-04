"""User-facing workflow properties independent of the R reference fixtures."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import softDOE


@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_linear_screening_tracks_factor_and_response_units(structure):
    design = softDOE.soft(4, 4, structure, m=16, seed=23, backend="cpu")["design"]
    coefficients = np.array([1.5, -3.0, 0.0, 8.0])
    response = design @ coefficients
    actual = softDOE.measure(design, response)
    np.testing.assert_allclose(actual["mustar"], np.abs(coefficients), atol=1e-14)
    assert actual["t"][2] == 0

    # Changing response units changes elementary effects but not importance t.
    transformed = softDOE.measure(design, -2.5 * response + 100)
    np.testing.assert_allclose(transformed["mustar"], 2.5 * actual["mustar"], atol=1e-13)
    np.testing.assert_allclose(transformed["t"], actual["t"], atol=1e-14)

    # Changing factor units affects only the corresponding elementary effects.
    factor_scales = np.array([2.0, 4.0, 8.0, 16.0])
    rescaled = softDOE.measure(design * factor_scales + 10, response)
    np.testing.assert_allclose(rescaled["mustar"], actual["mustar"] / factor_scales, atol=1e-14)
    np.testing.assert_allclose(rescaled["t"], actual["t"], atol=1e-14)


@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_screening_preserves_factor_labels_and_paired_row_order(structure):
    design = softDOE.soft(3, 4, structure, m=16, seed=51, backend="cpu")["design"]
    response = np.sin(design[:, 0] * design[:, 1]) + design[:, 2] ** 2
    original = softDOE.measure(design, response)
    columns = [2, 0, 1]
    # The same row permutation in every OFAT block preserves paired contrasts.
    rows = (np.arange(4)[:, None] * 4 + np.array([3, 0, 2, 1])).ravel()
    reordered = softDOE.measure(design[rows][:, columns], response[rows])
    for key in ("mustar", "t"):
        np.testing.assert_allclose(reordered[key], original[key][columns], atol=1e-14)


def test_measure_accepts_an_external_odd_base_run_count():
    # MOFAT screening designs can have odd l, although soft() requires even l.
    base = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    design = np.vstack((base, base + [0.2, 0], base + [0, 0.2]))
    actual = softDOE.measure(design.tolist(), (design @ [2.0, -5.0]).tolist())
    np.testing.assert_allclose(actual["mustar"], [2.0, 5.0], atol=1e-14)
    assert np.all(np.isfinite(actual["t"]))


@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_augmented_workflow_keeps_original_screening_rows(structure):
    base = softDOE.soft(3, 4, structure, m=16, seed=72, backend="cpu")["design"]
    saved = base.copy()
    base.setflags(write=False)
    augmented = softDOE.a_soft(n_aug=5, N=32, ofat_design=base, backend="cpu")["design"]
    np.testing.assert_array_equal(base, saved)
    np.testing.assert_array_equal(augmented[: len(base)], saved)
    response = augmented @ [2.0, 0.0, -3.0]
    effects = softDOE.measure(augmented[: len(base)], response[: len(base)])
    np.testing.assert_allclose(effects["mustar"], [2.0, 0.0, 3.0], atol=1e-14)


def test_q_is_invariant_to_row_order_without_mutating_inputs():
    rng = np.random.default_rng(15)
    design, points = rng.random((7, 3)), rng.random((13, 3))
    saved_design, saved_points = design.copy(), points.copy()
    design.setflags(write=False)
    points.setflags(write=False)
    original = softDOE.Q(1.5, 0.1, design, points, backend="cpu")
    reordered = softDOE.Q(1.5, 0.1, design[::-1], points[::-1], backend="cpu")
    for key in ("Q", "normalized_Q"):
        assert reordered[key] == pytest.approx(original[key], rel=2e-13)
    np.testing.assert_array_equal(design, saved_design)
    np.testing.assert_array_equal(points, saved_points)


def test_seeded_parallel_cpu_workflows_match_serial_without_changing_rng():
    def workflow(seed):
        structure = "strict" if seed % 2 else "standard"
        base = softDOE.soft(3, 4, structure, m=8, seed=seed, backend="cpu")["design"]
        return softDOE.a_soft(n_aug=2, N=16, ofat_design=base, backend="cpu")["design"]

    rng_before = np.random.get_state()
    expected = [workflow(seed) for seed in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(workflow, range(4)))
    for serial, parallel in zip(expected, actual):
        np.testing.assert_array_equal(parallel, serial)
    for before, after in zip(rng_before, np.random.get_state()):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("structure", ["standard", "strict"])
def test_smallest_generation_and_single_candidate_augmentation(structure):
    result = softDOE.soft(2, 2, structure, m=1, seed=6, backend="cpu")
    assert result["design"].shape == (6, 2)
    assert result["qmc_size"] == 5
    extended = softDOE.a_soft(2, 2, structure, n_aug=1, N=1, m=1,
                              seed=6, backend="cpu")
    assert extended["design"].shape == (7, 2)
    assert 0 < result["Q"] < 1
    assert 0 < extended["Q"] < 1
    effects = softDOE.measure(extended["design"][:6], extended["design"][:6, 0])
    np.testing.assert_allclose(effects["mustar"], [1.0, 0.0], atol=1e-14)


@pytest.mark.parametrize("problem", ["zero_step", "repeated_factor", "two_factors"])
def test_measure_rejects_invalid_ofat_contrasts(problem):
    base = np.array([[0.1, 0.2], [0.6, 0.7]])
    first = base + [0.2, 0]
    second = base + [0, 0.2]
    if problem == "zero_step":
        first[0] = base[0]
    elif problem == "repeated_factor":
        second = base + [0.3, 0]
    else:
        first += [0, 0.1]
    design = np.vstack((base, first, second))
    with pytest.raises(ValueError):
        softDOE.measure(design, np.arange(6, dtype=float))
