"""Batched strict candidates, retaining only the accepted CUDA Q state."""
from __future__ import annotations

import numpy as np

from ._engine import QState


class PairedUpdates:
    def __init__(self, parent: QState, left, right) -> None:
        self.parent = parent
        ops = parent.ops
        torch = ops.torch
        self.blocks = torch.stack((ops.array(left), ops.array(right)))
        k, p = self.blocks.shape[1:]
        self.k = k
        n, m = len(parent.design), len(parent.evaluation)
        flat = self.blocks.reshape(2 * k, p)
        cross = ops.correlation(parent.design, flat, parent.beta, alpha=parent.alpha)
        W = ops.solve(parent.L, cross).reshape(n, 2, k).permute(1, 0, 2)
        # Small block correlations are batched with the same bounded routine.
        diagonals = ops.correlation(flat, flat, parent.beta, alpha=parent.alpha).reshape(2, k, 2, k)
        diagonal = torch.stack((diagonals[0, :, 0, :], diagonals[1, :, 1, :]))
        indices = torch.arange(k, device=ops.device)
        diagonal[:, indices, indices] += parent.nugget
        self.W = W
        self.G = torch.linalg.cholesky(diagonal - W.transpose(1, 2) @ W)
        rhs = ops.correlation(flat, parent.evaluation, parent.beta, log=True,
                              alpha=parent.alpha).reshape(2, k, m)
        self.scales = torch.maximum(parent.scales, rhs.amax(dim=1))
        self.factors = (parent.scales - self.scales).exp()
        # Rescale after the matrix product; avoid two copies of the entire
        # parent solution matrix for candidates that may be rejected.
        residual = (rhs - self.scales[:, None, :]).exp() - (
            W.transpose(1, 2) @ parent.solutions) * self.factors[:, None, :]
        self.Z = torch.linalg.solve_triangular(self.G, residual, upper=False)
        self.norms = parent.norms * self.factors.square() + self.Z.square().sum(dim=1)
        self.logs = 2 * self.scales + ops.log_norm(self.norms)
        # One host readback checks both scores and all finite values.
        scores = self.logs.amin(dim=1)
        valid = torch.isfinite(self.logs).all().to(dtype=scores.dtype)
        values = ops.numpy(torch.cat((scores, valid.reshape(1))))
        if values[2] != 1 or not np.isfinite(values[:2]).all():
            raise FloatingPointError("Paired CUDA Q evaluation produced nonfinite values.")
        self.scores = tuple(float(x) for x in values[:2])

    def accept(self, index: int) -> QState:
        parent, k = self.parent, self.k
        ops = parent.ops
        n = len(parent.design)
        result = object.__new__(QState)
        result.ops, result.beta, result.nugget = ops, parent.beta, parent.nugget
        result.alpha = parent.alpha
        result.design = ops.concatenate((parent.design, self.blocks[index]))
        result.evaluation = parent.evaluation
        result.scales, result.norms = self.scales[index], self.norms[index]
        result.log_values = self.logs[index]
        result.solutions = ops.concatenate((parent.solutions * self.factors[index], self.Z[index]))
        result.L = ops.zeros((n + k, n + k))
        result.L[:n, :n] = parent.L
        result.L[n:, :n] = self.W[index].T
        result.L[n:, n:] = self.G[index]
        return result
