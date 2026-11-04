from __future__ import annotations

import math
import numpy as np

from ._backend import _run_with_fallback
from ._engine import ArrayOps, QState
from ._qmc import _sobol_points, _unique_rows_preserve_order
from ._soft import _dimensions, _prepare, _optimize
from ._validation import integer, matrix, structure_name


def _augment(design, candidates, n_aug: int, beta: float, ops: ArrayOps,
             return_score: bool = False):
    if ops.backend == "cpu" and getattr(ops, "native", False):
        from . import _native
        return _native.augment_cpu(design, candidates, n_aug, beta, return_score)
    if ops.backend == "metal" and getattr(ops, "native", False):
        from . import _native
        from ._metal import shader_source
        return _native.augment_metal(design, candidates, n_aug, beta, shader_source(), return_score)
    state = QState(design, candidates, beta, ops)
    eligible = np.ones(len(candidates), dtype=bool)
    selected_rows = []
    for step in range(n_aug):
        while np.any(eligible):
            values = ops.numpy(state.log_values).copy()
            values[~eligible] = np.inf
            selected = int(np.argmin(values))
            if ops.backend != "cpu":
                dtype = np.float32 if ops.backend == "metal" else np.float64
                close = eligible & (np.abs(values - values[selected]) <=
                                    np.sqrt(np.finfo(dtype).eps) * max(1.0, abs(values[selected])))
                if close.sum() > 1:
                    # Resolve close augmentation scores in double precision.
                    indices = np.flatnonzero(close)
                    original = np.vstack((design, candidates[selected_rows]))
                    precise = QState(original, candidates[indices], beta, ArrayOps())
                    selected = int(indices[np.argmin(precise.log_values)])
            eligible[selected] = False
            try:
                block = candidates[selected:selected + 1]
                if step + 1 == n_aug and not return_score:
                    state.append_factors(ops.array(block))
                    updated = state
                else:
                    updated = state.append(block)
            except np.linalg.LinAlgError:
                continue
            state = updated
            selected_rows.append(selected)
            break
        else:
            raise ValueError("No numerically eligible augmentation candidate remains.")
    # Preserve original double coordinates, including a supplied base.
    result = np.vstack((design, candidates[selected_rows]))
    return (result, state.score) if return_score else result


def a_soft(p=None, l=None, structure: str = "standard", n_aug=None,
           m: int = 500, N: int = 10000, backend: str = "auto",
           ofat_design=None, *, seed=None) -> dict:
    """Construct an augmented SOFT design, matching MOFAT's two-stage search.

    ``n_aug`` positive rows are selected sequentially from ``N`` Sobol
    candidates. With no ``ofat_design``, generate ``l*(p+1)`` initial rows
    using standard or strict SOFT. A supplied finite matrix in [0,1] is held
    fixed, overrides ``p``/``l``, and bypasses ``m``/``structure``/``seed``.
    Its row count must equal ``l*(p+1)`` for even ``l >= 2`` and ``p >= 2``.

    Candidate bounds are always [0.25/(2*l+n_aug), 1-0.25/(2*l+n_aug)].
    The initial SOFT search uses beta=(2*l)**-2; augmentation uses
    beta=(2*l+n_aug)**-2. Both use alpha=1 and nugget=1e-6.

    Returns ``design``, ``Q``, ``normalized_Q``, ``backend_used``, ``device``,
    and ``fallback_reason``. The scores evaluate the complete final design
    over all scaled Sobol candidates (including selected candidates), using
    the augmentation beta above. ``normalized_Q`` is Q**(1/p), computed in
    log space. GPU failure restarts the entire workflow on CPU.
    Use only the initial OFAT rows with :func:`measure`.

    Examples
    --------
    >>> result = a_soft(p=2, l=4, n_aug=8, seed=1, backend="cpu")
    >>> result["design"].shape
    (20, 2)
    >>> extended = a_soft(n_aug=8, ofat_design=result["design"][:12], backend="cpu")
    >>> extended["design"].shape
    (20, 2)
    """
    existing = None
    if ofat_design is not None:
        existing = matrix(ofat_design, "ofat_design").copy()
        rows, p = existing.shape
        l, remainder = divmod(rows, p + 1)
        if remainder or l < 2 or l % 2:
            raise ValueError("The OFAT design should contain l * (p + 1) runs for even l >= 2.")
        if np.any((existing < 0) | (existing > 1)):
            raise ValueError("ofat_design values must lie in [0, 1].")
    p, l = _dimensions(p, l)
    n_aug = integer(n_aug, "n_aug")
    N = integer(N, "N")
    if n_aug > N:
        raise ValueError("n_aug must not exceed N.")
    total_levels = 2 * l + n_aug
    boundary = 0.25 / total_levels
    prepared = None
    if existing is None:
        prepared = _prepare(p, l, integer(m, "m"), seed, total_levels)
        structure = structure_name(structure)
    candidates = _unique_rows_preserve_order(
        boundary + _sobol_points(N, p) * (1 - 2 * boundary)
    )
    if len(candidates) < n_aug:
        raise ValueError("The deduplicated candidate pool contains fewer than n_aug rows.")

    def workflow(ops):
        if existing is None:
            try:
                base, _ = _optimize(prepared, structure, ops)
            except Exception as error:
                error.soft_stage = "initial design generation"
                raise
        else:
            base = existing
        try:
            return _augment(base, candidates, n_aug, 1 / total_levels**2, ops,
                            return_score=True)
        except Exception as error:
            error.soft_stage = "augmentation"
            raise

    result, used, device, reason = _run_with_fallback(backend, workflow)
    design, score = result
    return dict(design=design, Q=math.exp(score), normalized_Q=math.exp(score / p),
                backend_used=used, device=device, fallback_reason=reason)
