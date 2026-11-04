"""Find the NVRTC runtime bundled with CUDA-enabled PyTorch installations."""
from functools import lru_cache
from pathlib import Path
from ctypes.util import find_library


@lru_cache(maxsize=None)
def configuration(device: int):
    from . import _native
    if not hasattr(_native, "optimize_cuda"):
        return None
    import torch
    root = Path(torch.__file__).resolve().parent
    candidates = sorted((root.parent / "nvidia").glob("**/libnvrtc.so*"))
    candidates += sorted((root / "lib").glob("libnvrtc*.so*"))
    candidates = [p for p in candidates if "builtins" not in p.name]
    library = str(candidates[0]) if candidates else find_library("nvrtc")
    if library is None:
        # CUDA tensor operations remain available without runtime compilation.
        return None
    major, minor = torch.cuda.get_device_capability(device)
    source = Path(__file__).with_name("_strict.cu").read_text()
    return source, library, device, 10 * major + minor
