"""Fused Metal scoring for strict SOFT's many small block updates.

Small correlation factorizations stay on the CPU in double precision. The
expensive solves against all evaluation points run in log-scaled float32 on
Metal, with reusable storage and one readback per pair of candidates.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ._engine import ArrayOps


@lru_cache(maxsize=1)
def _library():
    import torch
    return torch.mps.compile_shader(
        Path(__file__).with_name("_strict.metal").read_text(encoding="utf-8")
    )


class StrictMetalEvaluator:
    """Append-only scoring state; candidate evaluation does not mutate it."""

    def __init__(self, evaluation, beta: float, block_size: int,
                 max_rows: int, ops: ArrayOps) -> None:
        self.ops, self.cpu = ops, ArrayOps()
        self.library = _library()
        self.beta, self.k = float(beta), block_size
        self.m, self.p = evaluation.shape
        self.points = ops.array(np.ascontiguousarray(evaluation.T))
        empty = lambda shape: ops.torch.empty(shape, device="mps", dtype=ops.dtype)
        self.solutions = empty((max_rows, self.m))
        self.scales = empty(self.m)
        self.norms = empty(self.m)
        self.z = empty((2, self.k, self.m))
        self.new_scales = empty((2, self.m))
        self.new_norms = empty((2, self.m))
        self.logs = empty((2, self.m))
        self.L = np.zeros((max_rows, max_rows))
        self.design = np.empty((max_rows, self.p))

    def initialize(self, design) -> None:
        self.n = len(design)
        self.design[:self.n] = design
        corr = self.cpu.correlation(design, design, self.beta)
        corr.flat[::self.n + 1] += 1e-6
        factor = self.cpu.chol(corr)
        self.L[:self.n, :self.n] = factor
        self.library.initialize(
            self.solutions, self.scales, self.norms, self.logs, self.points,
            self.ops.array(design), self.ops.array(np.ascontiguousarray(factor)),
            self.n, self.p, self.m, self.beta, 1.0, threads=self.m,
        )

    def evaluate_pair(self, left, right) -> np.ndarray:
        n, k = self.n, self.k
        blocks = np.stack((left, right))
        weights, factors = [], []
        for block in blocks:
            cross = self.cpu.correlation(self.design[:n], block, self.beta)
            diagonal = self.cpu.correlation(block, block, self.beta)
            diagonal.flat[::k + 1] += 1e-6
            W = self.cpu.solve(self.L[:n, :n], cross)
            G = self.cpu.chol(diagonal - self.cpu.crossproduct(W, W))
            weights.append(W.T)
            factors.append(G)
        self.candidates = blocks, weights, factors
        # A single upload holds both candidates' small matrix coefficients.
        packed = np.concatenate((blocks.ravel(), np.asarray(weights).ravel(),
                                 np.asarray(factors).ravel()))
        self.library.evaluate_pair(
            self.logs, self.z, self.new_scales, self.new_norms,
            self.solutions, self.scales, self.norms, self.points,
            self.ops.array(packed), n, k, self.p, self.m, self.beta,
            threads=2 * self.m,
        )
        values = self.logs.cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("Strict Metal Q evaluation produced nonfinite values.")
        return values.min(axis=1).astype(float)

    def accept(self, index: int) -> None:
        n, k = self.n, self.k
        blocks, weights, factors = self.candidates
        self.design[n:n + k] = blocks[index]
        self.L[n:n + k, :n] = weights[index]
        self.L[:n, n:n + k] = 0
        self.L[n:n + k, n:n + k] = factors[index]
        self.library.accept(
            self.solutions, self.scales, self.norms, self.z,
            self.new_scales, self.new_norms, n, k, self.m, index,
            threads=self.m,
        )
        self.n += k
