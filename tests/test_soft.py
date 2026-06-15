import numpy as np
import pytest

import softDOE


def _assert_ofat_structure(design, p, l):
    blocks = [design[i * l : (i + 1) * l] for i in range(p + 1)]
    changed = []
    for block in blocks[1:]:
        diffs = [not np.allclose(block[:, j], blocks[0][:, j]) for j in range(p)]
        assert sum(diffs) == 1
        changed.append(diffs.index(True))
    assert sorted(changed) == list(range(p))


def test_standard_soft_shape_bounds_and_structure():
    design = softDOE.soft(p=3, l=4, m=8, seed=123)

    assert design.shape == (16, 3)
    assert np.all(design > 0)
    assert np.all(design < 1)
    _assert_ofat_structure(design, p=3, l=4)


def test_strict_soft_shape_and_bounds():
    design = softDOE.soft(p=3, l=4, structure="strict", m=8, seed=123)

    assert design.shape == (16, 3)
    assert np.all(design > 0)
    assert np.all(design < 1)


def test_seed_makes_standard_design_deterministic():
    first = softDOE.soft(p=3, l=4, m=8, seed=42)
    second = softDOE.soft(p=3, l=4, m=8, seed=42)

    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"p": 1, "l": 4}, "p must be at least 2"),
        ({"p": 3, "l": 3}, "l must be even"),
        ({"p": 3, "l": 4, "structure": "bad"}, "structure"),
    ],
)
def test_soft_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        softDOE.soft(**kwargs)
