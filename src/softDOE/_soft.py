from __future__ import annotations

import math
from itertools import permutations, product

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from . import _core


def soft(p, l, structure="standard", m=500, seed=None):
    """Generate a SOFT design.

    Parameters
    ----------
    p : int
        Number of factors. Must be at least 2.
    l : int
        Number of base runs. Must be even.
    structure : {"standard", "strict"}, default="standard"
        OFAT structure to generate.
    m : int, default=500
        Number of uniform Monte Carlo samples used to evaluate Q.
    seed : int or None, default=None
        Seed for reproducible Python-side random generation.
    """
    p = _as_int(p, "p")
    l = _as_int(l, "l")
    m = _as_int(m, "m")

    if p < 2:
        raise ValueError("p must be at least 2.")
    if l <= 0:
        raise ValueError("l must be positive.")
    if l % 2 != 0:
        raise ValueError("l must be even.")
    if m <= 0:
        raise ValueError("m must be positive.")
    if structure not in {"standard", "strict"}:
        raise ValueError('structure must be either "standard" or "strict".')

    rng = np.random.default_rng(seed)
    qmc = _qmc_generate(m, p, rng)
    a_info = _a_fun(p, l, seed)
    x = 0.25 / (2 * l)
    levels = np.arange(x, 1 - x + 1e-15, (1 - 2 * x) / (2 * l - 1))
    delta = levels[l] - levels[0]

    if structure == "standard":
        design, _ = _swap(
            qmc,
            a_info["A"],
            l,
            p,
            "standard",
            levels,
            delta,
            a_info["L"],
            a_info["mod"],
            a_info["quotient"],
        )
    else:
        design, _ = _swap(
            qmc,
            a_info["A"],
            l,
            p,
            "strict",
            levels,
            delta,
            a_info["L"],
            a_info["mod"],
            a_info["quotient"],
        )

    return design


def _as_int(value, name):
    """Convert an input to int while rejecting booleans and non-integer values."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        ans = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if ans != value:
        raise TypeError(f"{name} must be an integer.")
    return ans


def _qmc_generate(n, p, rng, threshold=10):
    """Generate uniform Monte Carlo points and representative corner points for Q evaluation."""
    qmc = rng.random((n, p))
    if p <= threshold:
        vertices = np.array(list(product([0.0, 1.0], repeat=p)), dtype=float)
    else:
        vertices = np.where(qmc < 0.5, 1.0, 0.0)
    return _unique_rows_preserve_order(np.vstack([qmc, vertices]))


def _unique_rows_preserve_order(x):
    """Remove duplicate rows from an array while keeping the first occurrence order."""
    seen = set()
    rows = []
    for row in x:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return np.array(rows, dtype=float)


def _a_fun(p, l, seed):
    """Generate the base A0 matrix and bookkeeping values used by the swap search."""
    L = math.factorial(l)
    if p < L:
        A = _maximin_lhd_levels(l, p, seed)
        quotient = None
        mod = None
    else:
        quotient = p // L
        mod = p % L
        P_l = np.array(list(permutations(range(1, l + 1))), dtype=float).T
        blocks = [P_l] * quotient
        if mod > 0:
            blocks.append(P_l[:, :mod])
        A = np.hstack(blocks)
    return {"A": A, "L": L, "quotient": quotient, "mod": mod}


def _maximin_lhd_levels(l, p, seed):
    """Generate an l by p maximin Latin hypercube and map it to R-style level indices."""
    from pyDOE import lhs

    try:
        design = lhs(
            n=p,
            samples=l,
            criterion="centermaximin",
            iterations=100,
            seed=seed,
        )
    except TypeError:
        state = np.random.get_state()
        if seed is not None:
            np.random.seed(seed)
        try:
            design = lhs(n=p, samples=l, criterion="centermaximin", iterations=100)
        finally:
            np.random.set_state(state)
    return design * l + 0.5


def _level_trans(design, l, levels):
    """Transform integer-like A0 levels into the actual continuous SOFT levels."""
    b = np.asarray(design, dtype=float).copy()
    lower = b < ((l + 1) / 2)
    upper = b > ((l + 1) / 2)
    b[lower] = 2 * b[lower] - 1
    b[upper] = 2 * b[upper]
    idx = np.rint(b).astype(int) - 1
    if np.any(idx < 0) or np.any(idx >= levels.size):
        raise ValueError("A matrix produced invalid level indices.")
    return levels[idx]


def _shift_factor(a, factor_idx, delta):
    """Create one OFAT block by shifting a single factor by delta."""
    b = a.copy()
    b[:, factor_idx] = np.where(b[:, factor_idx] < 0.5, b[:, factor_idx] + delta, b[:, factor_idx] - delta)
    return b


def _standard_ofat_constr(A, p, delta):
    """Construct the standard OFAT design from the base block A."""
    blocks = [A]
    for i in range(p):
        blocks.append(_shift_factor(A, i, delta))
    return np.vstack(blocks)


def _strict_ofat_constr(A, p, delta, qmc):
    """Construct the strict OFAT design by greedily choosing the better block order."""
    l = A.shape[0]
    alpha = 1.0
    beta = 1.0 / ((2 * l) ** 2)

    design_blocks = [A]
    design_blocks.append(_shift_factor(design_blocks[0], 0, delta))
    old_X = np.vstack(design_blocks)
    old_R = _core.corr_matrix(old_X, None, None, alpha, beta)
    L = cholesky(old_R + np.eye(old_R.shape[0]) * 1e-6, lower=True, check_finite=False)
    old_R_qmc = _core.q_matrix(qmc, old_X, None, None, alpha, beta)
    old_val = _q_min_value(old_R_qmc, L)

    for i in range(1, p):
        candidates = [
            _shift_factor(design_blocks[0], i, delta),
            _shift_factor(design_blocks[i], i, delta),
        ]
        opts = []
        for candidate in candidates:
            new_R = _core.corr_matrix(old_X, candidate, old_R, alpha, beta)
            L_new = cholesky(new_R + np.eye(new_R.shape[0]) * 1e-6, lower=True, check_finite=False)
            new_R_qmc = _core.q_matrix(qmc, old_X, old_R_qmc, candidate, alpha, beta)
            opts.append((_q_min_value(new_R_qmc, L_new), new_R, new_R_qmc, candidate))

        best_idx = int(np.argmax([opt[0] for opt in opts]))
        old_val, old_R, old_R_qmc, candidate = opts[best_idx]
        if best_idx == 0:
            design_blocks = [candidate] + design_blocks
        else:
            design_blocks.append(candidate)
        old_X = np.vstack(design_blocks)

    return np.vstack(design_blocks), old_val


def _q_min_value(new_R, L):
    """Compute the scalar Q objective, matching min(Q_fun_rcpp(...)$val) in R."""
    solved = solve_triangular(L, new_R, lower=True, check_finite=False)
    return float(np.min(np.sum(solved * solved, axis=0)))


def _swap(qmc, A, l, p, ofat_type, levels, delta, L, mod, quotient):
    """Run the deterministic swap search that improves the SOFT design objective."""
    alpha = 1.0
    beta = 1.0 / ((2 * l) ** 2)
    A = _level_trans(A, l, levels)

    if ofat_type == "standard":
        design = _standard_ofat_constr(A, p, delta)
        R = _core.corr_matrix(design, None, None, alpha, beta)
        chol = cholesky(R + np.eye(R.shape[0]) * 1e-6, lower=True, check_finite=False)
        old_val = _q_min_value(_core.q_matrix(qmc, design, None, None, alpha, beta), chol)
    else:
        design, old_val = _strict_ofat_constr(A, p, delta, qmc)

    p_0 = p if mod is None else mod
    p_1 = 0 if quotient is None else L * quotient

    if p_0 > 1:
        for i in range(l - 1):
            for j in range(i + 1, l):
                for k in range(1, p_0):
                    col = p_1 + k
                    A_new = A.copy()
                    A_new[[i, j], col] = A_new[[j, i], col]
                    if ofat_type == "standard":
                        design_new = _standard_ofat_constr(A_new, p, delta)
                        R_new = _core.corr_matrix(design_new, None, None, alpha, beta)
                        chol = cholesky(
                            R_new + np.eye(R_new.shape[0]) * 1e-6,
                            lower=True,
                            check_finite=False,
                        )
                        new_val = _q_min_value(
                            _core.q_matrix(qmc, design_new, None, None, alpha, beta),
                            chol,
                        )
                    else:
                        design_new, new_val = _strict_ofat_constr(A_new, p, delta, qmc)

                    if new_val > old_val:
                        A = A_new
                        design = design_new
                        old_val = new_val

    return design, old_val
