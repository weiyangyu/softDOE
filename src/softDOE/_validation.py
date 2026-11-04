"""Validation shared by the public design generators."""
from __future__ import annotations

import numbers
import numpy as np


def positive_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive finite scalar.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return float(value)


def integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be an integer.")
    if not np.isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be a finite integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def matrix(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.ndim != 2 or 0 in raw.shape or raw.dtype.kind not in "iuf"
            or not np.all(np.isfinite(raw))):
        raise ValueError(f"{name} must be a nonempty finite numeric matrix.")
    return np.asarray(raw, dtype=float)


def structure_name(value) -> str:
    # MOFAT silently defaults to standard for an invalid structure.
    return value if isinstance(value, str) and value in ("standard", "strict") else "standard"
