import numpy as np
import pytest

from softDOE import measure


def test_measure_matches_hand_calculation():
    design = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.75],
            [0.50, 0.25],
            [0.50, 0.75],
            [0.25, 0.50],
            [0.75, 0.50],
        ]
    )
    y = np.array([1.0, 3.0, 2.0, 4.0, 1.5, 3.5])

    result = measure(design, y)

    np.testing.assert_allclose(result["mustar"], [4.0, 2.0])
    expected_t = np.array([0.5, 0.125]) / np.var(y, ddof=1)
    np.testing.assert_allclose(result["t"], expected_t)


def test_measure_rejects_bad_shape():
    with pytest.raises(ValueError, match="l \\* \\(p \\+ 1\\)"):
        measure(np.zeros((5, 2)), np.zeros(5))


def test_measure_rejects_bad_response_length():
    with pytest.raises(ValueError, match="same length"):
        measure(np.zeros((6, 2)), np.zeros(5))
