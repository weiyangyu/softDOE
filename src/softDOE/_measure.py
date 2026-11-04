from __future__ import annotations

import numpy as np

from ._validation import matrix


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
    design = matrix(design, "design")
    y = np.asarray(y)
    if y.ndim != 1 or y.dtype.kind not in "iuf" or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite real numeric one-dimensional response vector.")
    y = np.asarray(y, dtype=float)

    n, p = design.shape
    if p < 1:
        raise ValueError("design must contain at least one factor.")
    if n % (p + 1) != 0:
        raise ValueError("design must have l * (p + 1) rows.")
    if y.shape[0] != n:
        raise ValueError("y must have the same length as design rows.")

    l = n // (p + 1)
    design_blocks = [design[i * l : (i + 1) * l, :] for i in range(p + 1)]
    y_blocks = np.split(y, p + 1)
    # Total effects are invariant to response scaling. A power-of-two scale
    # preserves significands while preventing overflow/underflow in squares.
    exponent = int(np.frexp(np.max(np.abs(y)))[1])
    scaled_y = np.ldexp(y, -exponent)
    scaled_blocks = np.split(scaled_y, p + 1)

    y_diff = [None] * p
    mustar = np.empty(p)

    for i in range(p):
        matrix_idx, factor_idx = _find_a(design_blocks, i + 1)
        if y_diff[factor_idx] is not None:
            raise ValueError("design repeats a factor instead of screening each factor once.")
        y_diff[factor_idx] = np.abs(scaled_blocks[i + 1] - scaled_blocks[matrix_idx])
        mustar[factor_idx] = _mean_elementary_effect(
            design_blocks[matrix_idx][:, factor_idx], design_blocks[i + 1][:, factor_idx],
            y_blocks[matrix_idx], y_blocks[i + 1],
        )

    if any(v is None for v in y_diff):
        raise ValueError("design does not have a valid one-factor-at-a-time block structure.")

    var_y = np.var(scaled_y, ddof=1)
    if var_y == 0:
        t = np.full(p, np.nan)
    else:
        vtot = np.array([np.sum(diff**2) for diff in y_diff], dtype=float) / (2 * l)
        t = vtot / var_y

    return {"mustar": mustar, "t": t}


def _scaled_difference(before, after):
    exponent = np.frexp(np.maximum(np.abs(before), np.abs(after)))[1]
    difference = np.abs(np.ldexp(after, -exponent) - np.ldexp(before, -exponent))
    return difference, exponent


def _mean_elementary_effect(x_before, x_after, y_before, y_after):
    # Scale each pair separately: unrelated large responses must not erase a
    # small contrast. Normalize the positive slopes before averaging so their
    # sum cannot overflow when their mean is representable.
    dy, y_exponent = _scaled_difference(y_before, y_after)
    dx, x_exponent = _scaled_difference(x_before, x_after)
    if np.any(dx == 0):
        raise ValueError("design contains zero OFAT step sizes.")
    mantissa, exponent = np.frexp(dy / dx)
    if not np.any(mantissa):
        return 0.0
    exponent += y_exponent - x_exponent
    largest = int(np.max(exponent[mantissa != 0]))
    return np.ldexp(np.mean(np.ldexp(mantissa, exponent - largest)), largest)


def _find_a(blocks, idx):
    b = blocks[idx]
    p = b.shape[1]
    for j in range(idx):
        a = blocks[j]
        with np.errstate(over="ignore"):
            col_same = np.array(
                [np.allclose(a[:, k], b[:, k], rtol=0.0, atol=1e-12) for k in range(p)]
            )
        changed = np.flatnonzero(~col_same)
        if changed.size == 1:
            return j, int(changed[0])
    raise ValueError("could not find the matching prior OFAT block.")
