"""Public MIM-kernel Q criterion, following MOFAT's Q interface."""
from __future__ import annotations

import math

from ._backend import _run_with_fallback
from ._engine import QState
from ._validation import matrix, positive_scalar


def _score(alpha, beta, design, qmc, ops) -> float:
    if ops.backend == "cpu" and getattr(ops, "native", False):
        from . import _native
        return float(_native.evaluate_cpu(design, qmc, beta, alpha=alpha).min())
    if ops.backend == "metal" and getattr(ops, "native", False):
        from . import _native
        from ._metal import shader_source
        return float(_native.evaluate_metal(design, qmc, beta, alpha, shader_source()))
    return QState(design, qmc, beta, ops, alpha=alpha).score


def Q(alpha, beta, design, qmc, backend: str = "auto") -> dict:
    """Evaluate a design's Q criterion using MOFAT's MIM kernel.

    ``alpha`` and ``beta`` must be positive finite scalars. ``design`` and
    ``qmc`` must be nonempty finite numeric matrices with the same number
    of columns. Points are used exactly as supplied; they need not lie in
    [0, 1], and no Sobol points are generated inside this function.

    With K(x, y) = prod_j (1 + (x_j-y_j)**2/beta)**(-alpha), compute
    min_z r(z).T @ (R + 1e-6*I)**(-1) @ r(z) over the rows of ``qmc``.
    Here R is the design correlation matrix and r(z) correlates the design
    with z. Larger Q indicates better coverage under this kernel and pool.

    Return ``Q``, ``normalized_Q`` (Q**(1/p)), and ``backend_used``.
    Normalization uses log Q, retaining precision when raw Q underflows.
    Backends are "auto", "cpu", "metal", and "cuda"; unavailable or failed
    GPU execution falls back to CPU. Metal uses float32 scoring with CPU
    double-precision factorization; CPU and CUDA use double precision.

    Examples
    --------
    >>> from softDOE import soft, qmc_generate, Q
    >>> design = soft(p=2, l=4, seed=1, backend="cpu")["design"]
    >>> scores = Q(1, 1/(2*4)**2, design, qmc_generate(n=1000, p=2), backend="cpu")
    >>> scores["backend_used"]
    'cpu'
    """
    alpha = positive_scalar(alpha, "alpha")
    beta = positive_scalar(beta, "beta")
    design, qmc = matrix(design, "design"), matrix(qmc, "qmc")
    if design.shape[1] != qmc.shape[1]:
        raise ValueError("design and qmc must have the same number of columns.")

    def workflow(ops):
        try:
            return _score(alpha, beta, design, qmc, ops)
        except Exception as error:
            error.soft_stage = "Q evaluation"
            raise

    score, used, _, _ = _run_with_fallback(backend, workflow)
    return dict(Q=math.exp(score), normalized_Q=math.exp(score / design.shape[1]),
                backend_used=used)
