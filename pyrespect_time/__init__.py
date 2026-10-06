"""
pyrespect_time
--------------
Extract continuous and discrete relaxation spectra from time-domain
G(t) data.

Public API
----------
    from pyrespect_time import ReSpect, ReSpectConfig

The public interface mirrors that of pyrespect_freq.
"""

from .config import ReSpectConfig, ReSpectError, ReSpectWarning
from .continuous import ContinuousResult
from .discrete import DiscreteResult
from .solver import ReSpect

__all__ = [
    "ReSpect",
    "ReSpectConfig",
    "ReSpectError",
    "ReSpectWarning",
    "ContinuousResult",
    "DiscreteResult",
]
__version__ = "2.1.0"
