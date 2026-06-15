from __future__ import annotations

import numpy as np


def measure(design, y):
    """Compute screening measures for a SOFT/Sobol'-structured design.

    Parameters
    ----------
    design : array_like, shape (l * (p + 1), p)
        Design matrix with one base block followed by one-factor-at-a-time
        blocks.
    y : array_like, shape (l * (p + 1),)
        Response vector.

    Returns
    -------
    dict
        ``{"mustar": array, "t": array}``, matching the R function output.
    """
    design = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if design.ndim != 2:
        raise ValueError("design must be a two-dimensional array.")

    n, p = design.shape
    if p < 1:
        raise ValueError("design must contain at least one factor.")
    if n % (p + 1) != 0:
        raise ValueError("design must have l * (p + 1) rows.")
    if y.shape[0] != n:
        raise ValueError("y must have the same length as design rows.")

    l = n // (p + 1)
    design_blocks = [design[i * l : (i + 1) * l, :] for i in range(p + 1)]
    y_blocks = [y[i * l : (i + 1) * l] for i in range(p + 1)]

    y_diff = [None] * p
    delta = [None] * p

    for i in range(p):
        matrix_idx, factor_idx = _find_a(design_blocks, i + 1)
        y_diff[factor_idx] = np.abs(y_blocks[i + 1] - y_blocks[matrix_idx])
        delta[factor_idx] = np.abs(
            design_blocks[i + 1][:, factor_idx]
            - design_blocks[matrix_idx][:, factor_idx]
        )

    if any(v is None for v in y_diff) or any(v is None for v in delta):
        raise ValueError("design does not have a valid one-factor-at-a-time block structure.")

    with np.errstate(divide="raise", invalid="raise"):
        try:
            mustar = np.array(
                [np.sum(diff / step) / l for diff, step in zip(y_diff, delta)],
                dtype=float,
            )
        except FloatingPointError as exc:
            raise ValueError("design contains zero OFAT step sizes.") from exc

    var_y = np.var(y, ddof=1)
    if var_y == 0:
        t = np.full(p, np.nan)
    else:
        vtot = np.array([np.sum(diff**2) for diff in y_diff], dtype=float) / (2 * l)
        t = vtot / var_y

    return {"mustar": mustar, "t": t}


def _find_a(blocks, idx):
    b = blocks[idx]
    p = b.shape[1]
    for j in range(idx):
        a = blocks[j]
        col_same = np.array(
            [np.allclose(a[:, k], b[:, k], rtol=0.0, atol=1e-12) for k in range(p)]
        )
        changed = np.flatnonzero(~col_same)
        if changed.size == 1:
            return j, int(changed[0])
    raise ValueError("could not find the matching prior OFAT block.")
