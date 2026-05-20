"""
config.py — Configuration dataclass for pyReSpect.

Supports direct instantiation, or loading from a TOML or YAML file:

    # Direct
    config = ReSpectConfig(ns=200, plateau=True)

    # From TOML
    config = ReSpectConfig.from_toml("inp.toml") where inp.toml contains

    [spectrum]
    plateau  = false

    [discrete]
    min_tau_spacing = 2.0
  
    # From YAML
    config = ReSpectConfig.from_yaml("inp.yaml")
"""

# Annotations introduced in Python 3.7. Very popular in typed Python code.
from __future__ import annotations

import warnings
from dataclasses import dataclass 
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# TOML support: stdlib in python 3.11+, tomli on 3.10 and below
# ---------------------------------------------------------------------------
try:
    import tomllib      # type: ignore
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

# YAML support: optional dependency
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Custom warning and exception classes
# ---------------------------------------------------------------------------

class ReSpectWarning(UserWarning):
    """Emitted when configuration values are valid but potentially inadvisable."""
    pass

class ReSpectError(Exception):
    """Raised for invalid configuration or runtime errors in pyReSpect."""
    pass

# ---------------------------------------------------------------------------
# Valid literal types
# ---------------------------------------------------------------------------

FreqEnd = Literal["lenient", "neutral", "strict"]

# Internal mapping from FreqEnd strings to the original integer codes
# for consistency with previous versions of pyReSpect
_FREQ_END_MAP: dict[str, int] = {
    "lenient": 1,
    "neutral": 2,
    "strict":  3,
}


# ---------------------------------------------------------------------------
# ReSpectConfig
# ---------------------------------------------------------------------------
@dataclass
class ReSpectConfig:
    """
    Configuration for a pyReSpect run.

    Parameters
    ----------
    ns : int
        Number of discretization points for the relaxation mode axis s.
        Default: 100.
    plateau : bool
        Whether to infer a non-zero plateau modulus G0. Default: False.
    freq_end : {"lenient", "neutral", "strict"}
        Controls how far the s axis extends beyond the t window.
        - "lenient"  : s extends to exp(±π/2) * t_min/t_max   [original: 1]
        - "neutral"  : s spans [t_min, t_max]                 [original: 2]
        - "strict"   : s contracts by exp(±π/2)               [original: 3]
        Default: "lenient".
    lam_min : float
        Lower bound of the regularization parameter search. Default: 1e-10.
    lam_max : float
        Upper bound of the regularization parameter search. Default: 1e3.
    lam_C : float or None
        Regularization parameter λ. If None, determined automatically via
        the Bayesian L-curve method. Default: None.
    lam_density : int
        Number of λ points per decade on the L-curve search grid. Default: 2.
    SmFacLam : float
        Smoothness factor in [-1, 1] that nudges λ_M up (> 0) or down (< 0)
        from the Bayesian optimum. 0.0 means no nudge. Default: 0.0.
    max_num_modes : int or None
        Maximum number of discrete Maxwell modes. None means no cap.
        Default: None.
    delta_base_weight_dist : float
        Step size for scanning the base weight distribution in (0, 1).
        Smaller values give a finer AIC search. Default: 0.2.
    min_tau_spacing : float
        Minimum ratio τ_{i+1}/τ_i below which adjacent modes are merged.
        Must be > 1.0. Default: 1.5.
    resample : bool
        Should I resample raw data by linear interpolation on log-grid?
        Useful when #datapoints is too large or unevenly dispersed Default: True.
    n_resample : int
        Number of points used when resampling input data onto a geometric
        grid (2-column input only). Default: 100.
    """

    # Continuous spectrum
    ns:                      int            = 100
    plateau:                 bool           = False
    freq_end:                FreqEnd        = "lenient"

    # Regularization
    lam_min:                 float          = 1e-10
    lam_max:                 float          = 1e3
    lam_C:                   Optional[float] = None
    lam_density:             int            = 2
    SmFacLam:                float          = 0.0

    # Discrete spectrum
    max_num_modes:           Optional[int]  = None
    delta_base_weight_dist:  float          = 0.2
    min_tau_spacing:         float          = 1.25

    # I/O
    resample:                bool           = True
    n_resample:              int            = 100

    # ------------------------------------------------------------------
    # Validation: some sanity checks on input settings
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()
        self._warn()

    def _validate(self) -> None:
        """Raise ValueError for clearly invalid parameter combinations."""

        if self.ns <= 0:
            raise ValueError(f"ns must be a positive integer, got {self.ns}.")

        if self.freq_end not in _FREQ_END_MAP:
            raise ValueError(
                f"freq_end must be one of {list(_FREQ_END_MAP)}, "
                f"got '{self.freq_end}'."
            )

        if self.lam_min <= 0:
            raise ValueError(f"lam_min must be > 0, got {self.lam_min}.")

        if self.lam_max <= 0:
            raise ValueError(f"lam_max must be > 0, got {self.lam_max}.")

        if self.lam_min >= self.lam_max:
            raise ValueError(
                f"lam_min must be < lam_max, "
                f"got lam_min={self.lam_min}, lam_max={self.lam_max}."
            )

        if self.lam_C is not None:
            if not (self.lam_min <= self.lam_C <= self.lam_max):
                raise ValueError(
                    f"lam_C={self.lam_C} must lie in "
                    f"[lam_min={self.lam_min}, lam_max={self.lam_max}]."
                )

        if self.lam_density < 1:
            raise ValueError(
                f"lam_density must be >= 1, got {self.lam_density}."
            )

        if not (-1.0 <= self.SmFacLam <= 1.0):
            raise ValueError(
                f"SmFacLam must be in [-1, 1], got {self.SmFacLam}."
            )

        if self.max_num_modes is not None and self.max_num_modes < 1:
            raise ValueError(
                f"max_num_modes must be >= 1 when specified, "
                f"got {self.max_num_modes}."
            )

        if not (0.0 < self.delta_base_weight_dist < 1.0):
            raise ValueError(
                f"delta_base_weight_dist must be in (0, 1), "
                f"got {self.delta_base_weight_dist}."
            )

        if self.min_tau_spacing <= 1.0:
            raise ValueError(
                f"min_tau_spacing must be > 1.0, got {self.min_tau_spacing}."
            )

        if self.n_resample <= 0:
            raise ValueError(
                f"n_resample must be a positive integer, got {self.n_resample}."
            )

    def _warn(self) -> None:
        """Emit ReSpectWarning for valid but potentially inadvisable settings."""

        if self.lam_density < 2:
            warnings.warn(
                "lam_density < 2: the L-curve grid is very coarse and may "
                "miss the optimal λ_M.",
                ReSpectWarning,
                stacklevel=3,
            )

        if self.ns < 50:
            warnings.warn(
                f"ns={self.ns} is small; the regularization problem may be "
                "poorly constrained. Consider ns >= 50.",
                ReSpectWarning,
                stacklevel=3,
            )

        if self.delta_base_weight_dist > 0.5:
            warnings.warn(
                f"delta_base_weight_dist={self.delta_base_weight_dist} is "
                "large; the AIC scan will be very coarse.",
                ReSpectWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @property
    def freq_end_int(self) -> int:
        """Return the integer code for freq_end (for internal kernel use)."""
        return _FREQ_END_MAP[self.freq_end]

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str) -> ReSpectConfig:
        """Load configuration from a TOML file.

        Parameters
        ----------
        path : str
            Path to the TOML file.

        Returns
        -------
        ReSpectConfig

        Example TOML layout::

            [spectrum]
            ns       = 100
            plateau  = false
            freq_end = "lenient"

            [regularization]
            lam_min     = 1e-10
            lam_max     = 1e3
            lam_density = 2
            SmFacLam    = 0.0

            [discrete]
            max_num_modes          = null
            delta_base_weight_dist = 0.2
            min_tau_spacing        = 1.5

            [io]
            n_resample = 100
        """
        if tomllib is None:
            raise ReSpectError(
                "TOML support requires Python >= 3.11, or 'tomli' to be "
                "installed: pip install tomli"
            )
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**_flatten(data))

    @classmethod
    def from_yaml(cls, path: str) -> ReSpectConfig:
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        ReSpectConfig
        """
        if yaml is None:
            raise ReSpectError(
                "YAML support requires 'pyyaml' to be installed: "
                "pip install pyyaml"
            )
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**_flatten(data))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _flatten(d: dict) -> dict:
    """Flatten a nested dict (one level deep) into a single dict.

    Sections like [spectrum], [regularization] etc. are merged into one
    flat mapping suitable for passing to ReSpectConfig as **kwargs.
    Top-level scalar keys are also preserved.
    """
    out: dict = {}
    for key, val in d.items():
        if isinstance(val, dict):
            out.update(val)
        else:
            out[key] = val
    return out
