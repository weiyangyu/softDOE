"""Native Metal discovery; public GPU calls do not require PyTorch on macOS."""
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def shader_source() -> str:
    return Path(__file__).with_name("_strict.metal").read_text()


class MetalOps:
    backend = "metal"
    native = True


def capabilities() -> dict:
    from . import _native
    if not hasattr(_native, "metal_info"):
        return dict(compiled=False, available=False, device=None,
                    reason="This build has no native Metal support.")
    info = _native.metal_info(shader_source())
    if not info["available"]:
        return info
    try:
        # Check full scoring, paired append, acceptance, and augmentation.
        base = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 0.7]])
        points = np.array([[0.25, 0.75, 0.4], [0.5, 0.5, 0.6], [0.1, 0.8, 0.9]])
        for strict in (False, True):
            design, score = _native.optimize_metal(base, points, 0.1, 0.1, 0, 0,
                                                   strict, shader_source())
            expected = _native.evaluate_cpu(design, points, 0.1).min()
            if not np.isclose(score, expected, rtol=1e-4, atol=1e-5):
                raise RuntimeError("Native Metal Q self-test disagrees with CPU.")
        actual = _native.augment_metal(base, points, 2, 0.1, shader_source())
        expected = _native.augment_cpu(base, points, 2, 0.1)
        if not np.array_equal(actual, expected):
            raise RuntimeError("Native Metal augmentation self-test disagrees with CPU.")
    except Exception as error:
        info.update(available=False, reason=f"Metal numerical self-test failed: {error}")
    return info
