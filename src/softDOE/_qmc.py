"""MOFAT's spacefillr Sobol sequence, including its float32 rounding.

The index permutation follows spacefillr 0.4.0 (MIT). Direction numbers
come from its PBRT/Joe--Kuo table. See THIRD_PARTY_NOTICES.txt.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np

from ._validation import integer


def _reverse_bits(x: np.ndarray) -> np.ndarray:
    for shift, mask in ((1, 0x55555555), (2, 0x33333333),
                        (4, 0x0F0F0F0F), (8, 0x00FF00FF), (16, 0x0000FFFF)):
        x = ((x >> shift) & np.uint32(mask)) | ((x & np.uint32(mask)) << shift)
    return x


@lru_cache(maxsize=1)
def _directions() -> np.ndarray:
    return np.load(Path(__file__).with_name("_sobol_directions.npy"), allow_pickle=False)


def _sobol_points(n: int, p: int) -> np.ndarray:
    """Match spacefillr::generate_sobol_set(n, p, seed=0) exactly."""
    if p > 1024:
        raise ValueError("p must not exceed 1024 (the MOFAT/spacefillr Sobol limit).")
    if n > 2**32:
        raise ValueError("n must not exceed 2**32 for the Sobol sequence.")
    # Hash the fixed seed, then apply the reversible index permutation.
    mask = 0xFFFFFFFF
    seed = 0x6217C6E1 ^ ((0xA14A177D * 0x9E3779B9) & mask)
    seed = ((seed ^ (seed >> 17)) * 0xED5AD4BB) & mask
    seed = ((seed ^ (seed >> 11)) * 0xAC4C1B51) & mask
    seed = ((seed ^ (seed >> 15)) * 0x31848BAB) & mask
    seed ^= seed >> 14
    index = _reverse_bits(np.arange(n, dtype=np.uint32))
    index ^= index * np.uint32(0x3D20ADEA)
    index += np.uint32(seed)
    index *= np.uint32((seed >> 16) | 1)
    index ^= index * np.uint32(0x05526C56)
    index ^= index * np.uint32(0x53A22864)
    index = _reverse_bits(index)
    directions = _directions()[:p]
    # A dimension at a time avoids an extra n*p array of uint32 values.
    points = np.empty((n, p), dtype=float)
    for dimension in range(p):
        bits = np.zeros(n, dtype=np.uint32)
        for bit in range(32):
            bits ^= ((index >> bit) & np.uint32(1)) * directions[dimension, bit]
        points[:, dimension] = np.minimum(
            bits.astype(np.float32) * np.float32(2.0**-32),
            np.float32(0.99999994),
        )
    return points


def _unique_rows_preserve_order(x: np.ndarray) -> np.ndarray:
    _, indices = np.unique(x, axis=0, return_index=True)
    return x[np.sort(indices)]


def qmc_generate(n: int, p: int, threshold: int = 10) -> np.ndarray:
    """Generate MOFAT-compatible Sobol points and unique corner points.

    ``n`` and ``p`` are positive integers; ``p <= 1024``. If ``p <= threshold``
    all ``2**p`` binary corners are added in R factorial order (first factor
    changes fastest). Otherwise representative opposite corners are added.
    Coordinates exactly equal to 0.5 stay at 0.5, as in MOFAT. The result is
    deterministic and its number of rows can differ from ``n``.

    Examples
    --------
    >>> qmc_generate(n=1000, p=2).shape
    (1004, 2)
    """
    n = integer(n, "n")
    p = integer(p, "p")
    threshold = integer(threshold, "threshold", 0)
    points = _sobol_points(n, p)
    if p <= threshold:
        if p >= 63 or (2**p) * p > 100_000_000:
            raise ValueError("The full corner set is too large; lower threshold.")
        vertices = ((np.arange(2**p, dtype=np.uint64)[:, None]
                     >> np.arange(p, dtype=np.uint64)) & 1).astype(float)
    else:
        vertices = np.where(points < 0.5, 1.0, np.where(points > 0.5, 0.0, points))
    return _unique_rows_preserve_order(np.vstack((points, vertices)))
