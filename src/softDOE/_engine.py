"""Stable Q scoring and incremental Cholesky updates on CPU or GPU."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.blas import dgemm

from . import _core


class ArrayOps:
    """Small adapter keeping factorization and solves on the chosen device."""

    def __init__(self, backend: str = "cpu") -> None:
        self.backend = backend
        self.native = True
        self.torch = None
        if backend != "cpu":
            import torch
            self.torch = torch
            self.device = "mps" if backend == "metal" else "cuda"
            self.dtype = torch.float32 if backend == "metal" else torch.float64

    def array(self, value):
        if self.torch is None:
            return np.asarray(value, dtype=float)
        return self.torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def numpy(self, value) -> np.ndarray:
        if self.torch is None:
            return np.asarray(value)
        return value.detach().cpu().numpy().astype(float)

    def concatenate(self, values, axis=0):
        if self.torch is None:
            return np.concatenate(values, axis=axis)
        return self.torch.cat(values, dim=axis)

    def zeros(self, shape):
        if self.torch is None:
            return np.zeros(shape)
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def exp(self, value):
        return np.exp(value) if self.torch is None else self.torch.exp(value)

    def scaled_exp(self, value, scales):
        # All correlations may have log=-inf for an extreme finite exponent.
        # Their criterion is zero; use a zero column instead of exp(-inf+inf).
        if self.torch is None:
            return np.exp(value - np.where(np.isneginf(scales), 0.0, scales))
        safe = self.torch.where(self.torch.isneginf(scales), 0.0, scales)
        return self.torch.exp(value - safe)

    def maximum(self, a, b):
        return np.maximum(a, b) if self.torch is None else self.torch.maximum(a, b)

    def max_columns(self, value):
        return value.max(axis=0) if self.torch is None else value.amax(dim=0)

    def norm_columns(self, value):
        return (value * value).sum(axis=0)

    def log_norm(self, value):
        if self.torch is None:
            return np.log(np.maximum(value, np.finfo(float).tiny))
        return self.torch.log(value.clamp_min(self.torch.finfo(self.dtype).tiny))

    def finite(self, value) -> bool:
        if self.torch is None:
            return bool(np.isfinite(value).all())
        return bool(self.torch.isfinite(value).all().item())

    def valid_log_scores(self, value) -> bool:
        # GPU arithmetic can overflow earlier than the stable CPU kernel.
        # Keep nonfinite GPU results on the existing CPU-fallback path.
        if self.backend != "cpu":
            return self.finite(value)
        if self.torch is None:
            return bool((np.isfinite(value) | np.isneginf(value)).all())
        return bool((self.torch.isfinite(value) | self.torch.isneginf(value)).all().item())

    def chol(self, value):
        if self.torch is None:
            return cholesky(value, lower=True, check_finite=False)
        return self.torch.linalg.cholesky(value)

    def solve(self, factor, rhs):
        if self.torch is None:
            return solve_triangular(factor, rhs, lower=True, check_finite=False)
        return self.torch.linalg.solve_triangular(factor, rhs, upper=False)

    def crossproduct(self, a, b):
        if self.torch is None:
            return dgemm(alpha=1.0, a=a, b=b, trans_a=True)
        return a.T @ b

    def correlation(self, x, y, beta: float, log: bool = False, alpha: float = 1.0):
        if self.torch is None:
            result = _core.q_matrix(y, x, None, None, alpha, beta)
            if not log:
                return result
            with np.errstate(divide="ignore"):
                answer = np.log(result)
            if np.any(result < np.finfo(float).tiny):
                # Recompute underflowed columns in log space.
                columns = np.any(result < np.finfo(float).tiny, axis=0)
                answer[:, columns] = _core._log_q_matrix(y[columns], x, alpha, beta)
            return answer
        # Accumulate log correlations without an n*m*p temporary tensor.
        if self.backend == "cuda":
            # Batch dimensions to reduce CUDA launch overhead. Bound the
            # temporary tensors even for large augmentation candidate pools.
            width = min(x.shape[1], 16)
            columns = max(1, (1 << 20) // max(1, len(x) * width))
            answer = self.zeros((len(x), len(y)))
            for start in range(0, len(y), columns):
                stop = min(start + columns, len(y))
                tile = answer[:, start:stop]
                for factor in range(0, x.shape[1], width):
                    difference = (x[:, None, factor:factor + width] -
                                  y[None, start:stop, factor:factor + width])
                    difference.square_().div_(beta).log1p_()
                    tile -= difference.sum(dim=2)
            if alpha != 1.0:
                answer *= alpha
            return answer if log else answer.exp()
        answer = self.zeros((len(x), len(y)))
        for factor in range(x.shape[1]):
            difference = x[:, factor, None] - y[None, :, factor]
            answer -= self.torch.log1p(difference.square() / beta)
        if alpha != 1.0:
            answer *= alpha
        return answer if log else answer.exp()


class QState:
    """Scaled correlation solves; appending a block costs a Schur update."""

    def __init__(self, design, evaluation, beta: float,
                 ops: ArrayOps, nugget: float = 1e-6, *, alpha: float = 1.0) -> None:
        self.ops, self.beta, self.nugget = ops, beta, nugget
        self.alpha = alpha
        self.design, self.evaluation = ops.array(design), ops.array(evaluation)
        corr = ops.correlation(self.design, self.design, beta, alpha=alpha)
        indices = np.arange(len(self.design))
        corr[indices, indices] += nugget
        self.L = ops.chol(corr)
        log_rhs = ops.correlation(self.design, self.evaluation, beta, log=True, alpha=alpha)
        self.scales = ops.max_columns(log_rhs)
        self.solutions = ops.solve(self.L, ops.scaled_exp(log_rhs, self.scales))
        self.norms = ops.norm_columns(self.solutions)
        self._update_values()

    def _update_values(self) -> None:
        with np.errstate(over="ignore"):
            self.log_values = 2 * self.scales + self.ops.log_norm(self.norms)
        if not self.ops.valid_log_scores(self.log_values):
            raise FloatingPointError("Q evaluation produced nonfinite values.")

    @property
    def score(self) -> float:
        return float(self.log_values.min().item())

    def append(self, block) -> QState:
        ops = self.ops
        block = ops.array(block)
        W, G = self.append_factors(block)
        log_rhs = ops.correlation(block, self.evaluation, self.beta, log=True, alpha=self.alpha)
        scales = ops.maximum(self.scales, ops.max_columns(log_rhs))
        factors = ops.scaled_exp(self.scales, scales)
        solutions = self.solutions * factors
        Z = ops.solve(G, ops.scaled_exp(log_rhs, scales) - ops.crossproduct(W, solutions))
        result = object.__new__(QState)
        result.ops, result.beta, result.nugget = ops, self.beta, self.nugget
        result.alpha = self.alpha
        result.design = ops.concatenate((self.design, block))
        result.evaluation = self.evaluation
        result.scales = scales
        result.norms = self.norms * factors**2 + ops.norm_columns(Z)
        result.solutions = ops.concatenate((solutions, Z))
        n, k = len(self.design), len(block)
        result.L = ops.zeros((n + k, n + k))
        result.L[:n, :n] = self.L
        result.L[n:, :n] = W.T
        result.L[n:, n:] = G
        result._update_values()
        return result

    def append_factors(self, block):
        """Factor a candidate without calculating Q for unused final updates."""
        ops = self.ops
        cross = ops.correlation(self.design, block, self.beta, alpha=self.alpha)
        diagonal = ops.correlation(block, block, self.beta, alpha=self.alpha)
        indices = np.arange(len(block))
        diagonal[indices, indices] += self.nugget
        W = ops.solve(self.L, cross)
        G = ops.chol(diagonal - ops.crossproduct(W, W))
        return W, G


def gpu_close(a: float, b: float, ops: ArrayOps) -> bool:
    """MOFAT's relative tolerance for a close GPU search decision."""
    if ops.backend == "cpu":
        return False
    # MOFAT's Metal and CUDA optimizers both use float32 epsilon. Retain that
    # decision tolerance even though this package's CUDA arithmetic is float64.
    return abs(a - b) <= np.sqrt(np.finfo(np.float32).eps) * max(1.0, abs(a), abs(b))


def cpu_score(design, evaluation, beta: float) -> float:
    return QState(design, evaluation, beta, ArrayOps()).score
