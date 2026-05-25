"""
solver.py — The primary public API for pyReSpect.

The ReSpect class is the single entry point for users. It orchestrates
data loading, continuous and discrete spectrum fitting, file output,
and plotting.

Typical usage
-------------
    from pyrespect import ReSpect, ReSpectConfig

    # Default configuration
    solver = ReSpect()
    solver.fit("Gt.dat")

    # Custom configuration, data as arrays
    config = ReSpectConfig(ns=200, plateau=True, freq_end="neutral")
    solver = ReSpect(config)
    solver.fit(t, Gt)

    # Access results
    print(solver.continuous.H)
    print(solver.discrete.tau)

    # Save and plot
    solver.save("output/", which=["base"]) or which="full" for diagnostics
    figs = solver.plot(which=["base"], toFile=True, path="output/")
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .config import ReSpectConfig, ReSpectError
from .continuous import ContinuousResult, fit_continuous
from .discrete import DiscreteResult, fit_discrete
from .io import load_data, save as _save
from .plotting import plot as _plot


class ReSpect:
    """Continuous and discrete relaxation spectrum solver.

    Parameters
    ----------
    config : ReSpectConfig or None, optional
        Configuration object. If None, default ReSpectConfig() is used.

    Attributes
    ----------
    continuous : ContinuousResult or None
        Results from the continuous spectrum fit. None until fit() is called.
    discrete : DiscreteResult or None
        Results from the discrete spectrum fit. None until fit() is called.
    t : np.ndarray or None
        Time points used in the fit. None until fit() is called.
    Gt : np.ndarray or None
        Experimental G(t) used in the fit. None until fit() is called.
    weights : np.ndarray or None
        Per-datapoint weights used in the fit. None until fit() is called.
    config : ReSpectConfig
        The active configuration.
    """

    def __init__(self, config: Optional[ReSpectConfig] = None) -> None:
        self.config:     ReSpectConfig             = config or ReSpectConfig()
        self.continuous: Optional[ContinuousResult] = None
        self.discrete:   Optional[DiscreteResult]   = None
        self.t:          Optional[np.ndarray]        = None
        self.Gt:         Optional[np.ndarray]        = None
        self.weights:    Optional[np.ndarray]        = None

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def fit(
        self,
        source:   Union[str, np.ndarray, tuple],
        Gt:       Optional[np.ndarray] = None,
        weights:  Optional[np.ndarray] = None,
        resample: bool                 = True,
    ) -> "ReSpect":
        """Fit the continuous and discrete relaxation spectra.

        Runs fit_continuous() followed by fit_discrete(), storing results
        as attributes. The continuous spectrum must be computed before the
        discrete spectrum; this ordering is enforced internally.

        Parameters
        ----------
        source : str, np.ndarray, or (np.ndarray, np.ndarray)
            The experimental data. Accepted forms:

            - ``"Gt.dat"``        : path to a 2- or 3-column data file.
            - ``t``               : 1-D numpy array of time points,
                                    in which case Gt must also be supplied.
            - ``(t, Gt)``         : tuple of 1-D numpy arrays.

        Gt : np.ndarray or None, optional
            Relaxation modulus array. Required when source is a 1-D numpy
            array of time points. Ignored otherwise.
        weights : np.ndarray or None, optional
            Per-datapoint weights. Optional for all input forms.
            If None, weights default to 1.0 (or are read from a 3-column
            file).
        resample : bool, optional
            If True (default), resample 2-column file or array data onto
            a geometric grid of config.n_resample points. Has no effect
            on 3-column file data.

        Returns
        -------
        self : ReSpect
            Returns self to allow method chaining:
            ``solver.fit("Gt.dat").save("output/").plot()``.

        Raises
        ------
        ReSpectError
            If source is a 1-D array but Gt is not supplied, or if the
            data cannot be loaded.
        """
        # Normalise source into something load_data understands
        source = self._normalise_source(source, Gt)

        # Load data
        self.t, self.Gt, self.weights = load_data(
            source,
            weights=weights,
            resample=resample,
            n_resample=self.config.n_resample,
        )

        # Stage 1: continuous spectrum
        self.continuous = fit_continuous(
            self.t, self.Gt, self.weights, self.config
        )

        # Stage 2: discrete spectrum (depends on continuous result)
        self.discrete = fit_discrete(
            self.t, self.Gt, self.weights, self.continuous, self.config
        )

        return self

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def save(
        self,
        path:  str,
        which: Union[str, list[str]] = "base",
    ) -> "ReSpect":
        """Write results to files in the specified output directory.

        Parameters
        ----------
        path : str
            Output directory. Created if it does not exist.
        which : str or list of str, optional
            Which outputs to write. Valid values:

            - ``"base"`` : H.dat, dmodes.dat, and Gfit.dat.
                           Gfit.dat has three columns: t, G_continuous,
                           G_discrete.
            - ``"full"`` : above + rho-eta.dat, logPlam.dat, Hlam.dat,
                           aic.dat.

            Default: ``"base"``.

        Returns
        -------
        self : ReSpect
            Returns self to allow method chaining.

        Raises
        ------
        ReSpectError
            If a requested output requires a result that has not yet
            been computed (i.e. fit() has not been called), or if an
            invalid 'which' token is supplied.
        """
        _save(
            path=path,
            which=which,
            t=self._require_data("save"),
            cont_result=self.continuous,
            disc_result=self.discrete,
        )
        return self

    def plot(
        self,
        which:  Union[str, list[str]] = "base",
        toFile: bool                  = False,
        path:   str                   = "./",
    ) -> list:
        """Plot the fitted spectra.

        Parameters
        ----------
        which : str or list of str, optional
            Which plots to produce. Valid values:

            - ``"base"`` : three-panel figure — H(s), g_i/τ_i overlay,
                           G(t) data vs continuous and discrete fits.
            - ``"full"`` : above + diagnostic figure — log p(λ), ρ-η
                           L-curve, and AIC scan.

            Default: ``"base"``.
        toFile : bool, optional
            If True, save figures to path in addition to returning them.
            Default: False.
        path : str, optional
            Output directory for figures when toFile=True.
            Default: ``"./"``.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            All figures produced, in order.

        Raises
        ------
        ReSpectError
            If a requested plot requires a result that has not yet been
            computed, or if an invalid 'which' token is supplied.
        """
        return _plot(
            which=which,
            toFile=toFile,
            path=path,
            t=self._require_data("plot"),
            Gt=self.Gt,
            weights=self.weights,
            cont_result=self.continuous,
            disc_result=self.discrete,
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str) -> "ReSpect":
        """Construct a ReSpect solver from a TOML configuration file.

        Parameters
        ----------
        path : str
            Path to the TOML configuration file.

        Returns
        -------
        ReSpect
        """
        return cls(ReSpectConfig.from_toml(path))

    @classmethod
    def from_yaml(cls, path: str) -> "ReSpect":
        """Construct a ReSpect solver from a YAML configuration file.

        Parameters
        ----------
        path : str
            Path to the YAML configuration file.

        Returns
        -------
        ReSpect
        """
        return cls(ReSpectConfig.from_yaml(path))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.continuous is not None
        return (
            f"ReSpect("
            f"fitted={fitted}, "
            f"ns={self.config.ns}, "
            f"plateau={self.config.plateau}, "
            f"freq_end='{self.config.freq_end}')"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_source(
        self,
        source: Union[str, np.ndarray, tuple],
        Gt:     Optional[np.ndarray],
    ) -> Union[str, tuple]:
        """Normalise the source argument into str or (t, Gt) tuple.

        Handles three calling conventions:
            solver.fit("Gt.dat")
            solver.fit(t, Gt)
            solver.fit((t, Gt))
        """
        if isinstance(source, str):
            return source
        elif isinstance(source, tuple):
            return source
        elif isinstance(source, np.ndarray):
            if Gt is None:
                raise ReSpectError(
                    "When source is a numpy array of time points, "
                    "Gt must also be supplied as the second argument."
                )
            return (source, np.asarray(Gt, dtype=float))
        else:
            raise ReSpectError(
                "source must be a file path (str), a tuple (t, Gt), "
                "or a numpy array of time points."
            )

    def _require_data(self, caller: str) -> np.ndarray:
        """Return self.t or raise ReSpectError if fit() hasn't been called."""
        if self.t is None:
            raise ReSpectError(
                f"Cannot call {caller}() before fit(). "
                "Run solver.fit(source) first."
            )
        return self.t
