"""Space-filling one-factor-at-a-time designs."""

from ._measure import measure
from ._soft import soft
from ._augment import a_soft
from ._qmc import qmc_generate
from ._q import Q
from ._backend import backend_info

__version__ = "0.2.0"

__all__ = ["soft", "a_soft", "measure", "qmc_generate", "Q", "backend_info"]
