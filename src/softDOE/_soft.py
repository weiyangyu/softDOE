from __future__ import annotations

import inspect
import math
from itertools import permutations
import numpy as np

from ._backend import _run_with_fallback
from ._engine import ArrayOps, QState, gpu_close
from ._qmc import qmc_generate
from ._validation import integer, structure_name


def soft(p: int, l: int, structure: str = "standard", m: int = 500,
         seed=None, backend: str = "auto") -> dict:
    """Generate a space-filling one-factor-at-a-time design, as in MOFAT.

    Parameters
    ----------
    p : int
        Number of factors, at least 2 (at most 1024).
    l : int
        Even number of base runs, at least 2.
    structure : {"standard", "strict"}
        OFAT structure; invalid values default to standard, matching MOFAT.
    m : int
        Sobol evaluation points before corner augmentation, default 500.
    seed : int or None
        Seed for PyDOE's initial centered maximin Latin hypercube. Sobol points always use
        MOFAT's fixed sequence. Equal R/Python seeds do not imply equal bases.
    backend : {"auto", "cpu", "cuda", "metal"}
        Requested device. Invalid, unavailable or failed GPU requests fall
        back to CPU; the returned diagnostics describe what actually ran.

    Returns
    -------
    dict
        ``design`` (shape ``(l*(p+1), p)``), ``Q``, ``normalized_Q``,
        ``qmc_size``, ``backend_used``, ``device``, ``fallback_reason``.
        ``normalized_Q`` is computed in log space even when ``Q`` underflows.

    Notes
    -----
    Since version 0.2, use ``soft(...)['design']`` to retrieve the array.
    Kernel parameters are alpha=1, beta=(2*l)**-2, nugget=1e-6.

    Examples
    --------
    >>> result = soft(p=2, l=4, seed=1, backend="cpu")
    >>> result["design"].shape
    (12, 2)
    """
    p, l = _dimensions(p, l)
    m = integer(m, "m")
    prepared = _prepare(p, l, m, seed, total_levels=2 * l)
    structure = structure_name(structure)
    result, used, device, reason = _run_with_fallback(
        backend, lambda ops: _optimize(prepared, structure, ops)
    )
    design, log_q = result
    return dict(design=design, Q=float(np.exp(log_q)),
                normalized_Q=float(np.exp(log_q / p)),
                qmc_size=len(prepared["qmc"]), backend_used=used,
                device=device, fallback_reason=reason)


def _dimensions(p, l) -> tuple:
    p = integer(p, "p", 2)
    l = integer(l, "l", 2)
    if l % 2:
        raise ValueError("l must be even.")
    if p > 1024:
        raise ValueError("p must not exceed 1024 (the MOFAT/spacefillr Sobol limit).")
    return p, l


def _prepare(p: int, l: int, m: int, seed, total_levels: int) -> dict:
    qmc = qmc_generate(m, p)
    info = _a_fun(p, l, seed)
    boundary = 0.25 / total_levels
    levels = np.linspace(boundary, 1 - boundary, 2 * l)
    return dict(p=p, l=l, qmc=qmc, A=_level_trans(info["A"], l, levels),
                delta=levels[l] - levels[0], beta=1 / (2 * l)**2,
                remainder=info["mod"], quotient=info["quotient"], factorial_l=info["L"])


def _a_fun(p: int, l: int, seed=None) -> dict:
    """Base level matrix: a maximin LHD, or replicated full permutations."""
    L = math.factorial(l)
    if p < L:
        A = _maximin_lhd_levels(l, p, seed)
        quotient = mod = None
    else:
        quotient, mod = divmod(p, L)
        perms = np.array(list(permutations(range(1, l + 1))), dtype=float).T
        A = np.tile(perms, (1, quotient))
        if mod:
            A = np.hstack((A, perms[:, :mod]))
    return {"A": A, "L": L, "quotient": quotient, "mod": mod}


def _maximin_lhd_levels(l: int, p: int, seed=None) -> np.ndarray:
    """Use PyDOE's centered maximin LHD across old/new import spellings."""
    try:
        from pydoe import lhs
    except ModuleNotFoundError as error:
        # Only retry a missing top-level module. Do not hide a broken PyDOE
        # dependency or create aliases in sys.modules.
        if error.name != "pydoe":
            raise
        from pyDOE import lhs

    kwargs = dict(n=p, samples=l, criterion="centermaximin", iterations=100)
    parameters = inspect.signature(lhs).parameters
    if "seed" in parameters:
        design = lhs(**kwargs, seed=seed)
    elif "random_state" in parameters:
        design = lhs(**kwargs, random_state=seed)
    else:
        # Older PyDOE uses NumPy's global RNG. Restore the caller's state.
        state = np.random.get_state()
        try:
            # An unseeded call must receive fresh entropy even though the
            # legacy implementation's global random state is restored below.
            local_seed = seed if seed is not None else np.random.SeedSequence().generate_state(1)[0]
            np.random.seed(local_seed)
            design = lhs(**kwargs)
        finally:
            np.random.set_state(state)
    return np.asarray(design, dtype=float) * l + 0.5


def _level_trans(design, l: int, levels: np.ndarray) -> np.ndarray:
    b = np.asarray(design, dtype=float).copy()
    lower = b < ((l + 1) / 2)
    upper = b > ((l + 1) / 2)
    b[lower] = 2 * b[lower] - 1
    b[upper] = 2 * b[upper]
    idx = np.rint(b).astype(int) - 1
    if np.any(idx < 0) or np.any(idx >= levels.size):
        raise ValueError("A matrix produced invalid level indices.")
    return levels[idx]


def _shift_factor(a: np.ndarray, factor_idx: int, delta: float) -> np.ndarray:
    b = a.copy()
    b[:, factor_idx] = np.where(b[:, factor_idx] < 0.5,
                                b[:, factor_idx] + delta, b[:, factor_idx] - delta)
    return b


def _standard_ofat_constr(A, p: int, delta: float) -> np.ndarray:
    return np.vstack([A] + [_shift_factor(A, i, delta) for i in range(p)])


def _initial(A, prepared: dict, structure: str, ops: ArrayOps, metal=None) -> tuple:
    p, delta, beta, qmc = (prepared[k] for k in ("p", "delta", "beta", "qmc"))
    if structure == "standard":
        design = _standard_ofat_constr(A, p, delta)
        return design, QState(design, qmc, beta, ops).score
    blocks = [A, _shift_factor(A, 0, delta)]
    if metal is not None:
        metal.initialize(np.vstack(blocks))
        for factor in range(1, p):
            left = _shift_factor(blocks[0], factor, delta)
            right = _shift_factor(blocks[-1], factor, delta)
            scores = metal.evaluate_pair(left, right)
            # MOFAT Metal keeps the left candidate when scores are close.
            index = 0 if scores[0] >= scores[1] or gpu_close(*scores, ops) else 1
            metal.accept(index)
            if index == 0:
                blocks.insert(0, left)
            else:
                blocks.append(right)
        return np.vstack(blocks), float(scores[index])
    state = QState(np.vstack(blocks), qmc, beta, ops)
    for factor in range(1, p):
        left = _shift_factor(blocks[0], factor, delta)
        right = _shift_factor(blocks[-1], factor, delta)
        if ops.backend == "cuda" and getattr(ops, "native", False):
            from ._cuda import PairedUpdates
            updates = PairedUpdates(state, left, right)
            scores = updates.scores
            index = 0 if scores[0] >= scores[1] or gpu_close(*scores, ops) else 1
            if index == 0:
                blocks.insert(0, left)
            else:
                blocks.append(right)
            if factor + 1 == p:
                return np.vstack(blocks), scores[index]
            state = updates.accept(index)
            continue
        left_state, right_state = state.append(left), state.append(right)
        left_score, right_score = left_state.score, right_state.score
        # Both MOFAT GPU backends choose left on a close score. CPU keeps
        # its original exact comparison; no GPU candidate is rescored on CPU.
        if left_score >= right_score or gpu_close(left_score, right_score, ops):
            blocks.insert(0, left)
            state = left_state
        else:
            blocks.append(right)
            state = right_state
    return np.vstack(blocks), state.score


def _optimize(prepared: dict, structure: str, ops: ArrayOps) -> tuple:
    p0 = prepared["p"] if prepared["remainder"] is None else prepared["remainder"]
    p1 = 0 if prepared["quotient"] is None else prepared["factorial_l"] * prepared["quotient"]
    if ops.backend == "cpu" and getattr(ops, "native", False):
        from . import _native
        return _native.optimize_cpu(prepared["A"], prepared["qmc"], prepared["beta"],
                                    prepared["delta"], p0, p1, structure == "strict")
    if ops.backend == "metal" and getattr(ops, "native", False):
        from . import _native
        from ._metal import shader_source
        return _native.optimize_metal(prepared["A"], prepared["qmc"], prepared["beta"],
                                      prepared["delta"], p0, p1, structure == "strict",
                                      shader_source())
    if (structure == "strict" and ops.backend == "cuda"
            and getattr(ops, "native", False) and getattr(ops, "device", None) == "cuda"):
        from ._cuda_native import configuration
        config = configuration(ops.torch.cuda.current_device())
        if config is not None:
            from . import _native
            return _native.optimize_cuda(prepared["A"], prepared["qmc"], prepared["beta"],
                                         prepared["delta"], p0, p1, *config)
    A = prepared["A"].copy()
    metal = None
    if structure == "strict" and ops.backend == "metal":
        from ._strict_metal import StrictMetalEvaluator
        metal = StrictMetalEvaluator(prepared["qmc"], prepared["beta"],
                                     prepared["l"], prepared["l"] * (prepared["p"] + 1), ops)
    current = _initial(A, prepared, structure, ops, metal)
    for i in range(prepared["l"] - 1):
        for j in range(i + 1, prepared["l"]):
            for k in range(1, p0):
                column = p1 + k
                candidate = A.copy()
                candidate[[i, j], column] = candidate[[j, i], column]
                proposed = _initial(candidate, prepared, structure, ops, metal)
                old_score, new_score = current[1], proposed[1]
                # Keep the incumbent on GPU ties, for both structures.
                if new_score > old_score and not gpu_close(old_score, new_score, ops):
                    A, current = candidate, proposed
    return current
