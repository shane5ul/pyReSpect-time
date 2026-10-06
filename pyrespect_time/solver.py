"""
solver.py — ReSpect: the primary user-facing solver class.

    from pyrespect_time import ReSpect, ReSpectConfig

    solver = ReSpect()                      # or ReSpect(ReSpectConfig(...))
    solver.fit("Gt.dat")                    # or solver.fit(t, Gt)

    solver.continuous.H        # continuous spectrum H(s)
    solver.continuous.G_fit    # predicted G(t) from the CRS
    solver.discrete.tau        # discrete relaxation times
    solver.discrete.G_fit      # predicted G(t) from the DRS

    solver.save(which="full", path="output/")
    figs = solver.plot(which="base")

fit() and save() return the solver, so they can be chained:

    ReSpect().fit("Gt.dat").save(which="base", path="output/")

The interface is the same as that of pyrespect_freq.ReSpect; only the
data passed to fit() differ.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import Optional, Sequence, Union

import numpy as np

from .config import ReSpectConfig, ReSpectError
from .continuous import ContinuousResult, fit_continuous
from .discrete import DiscreteResult, fit_discrete
from .io import load_data, save as _save, _VALID_WHICH
from .plotting import plot as _plot


def _accept_legacy_save_order(method):
    """Accept the pre-2.1 positional call ``save(path[, which])``.

    Before version 2.1 the signature was ``save(path, which="base")``. A
    first positional argument that is not a valid 'which' token is
    therefore taken to be the output directory, with a DeprecationWarning.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if (
            args
            and isinstance(args[0], (str, os.PathLike))
            and args[0] not in _VALID_WHICH
        ):
            warnings.warn(
                "save(path, which) is deprecated; the argument order is now "
                "save(which, path), as in pyrespect_freq. Use keywords: "
                "save(which='base', path='output/').",
                DeprecationWarning,
                stacklevel=2,
            )
            which = args[1] if len(args) > 1 else kwargs.get("which", "base")
            return method(self, which=which, path=args[0])
        return method(self, *args, **kwargs)

    return wrapper


class ReSpect:
    """Solver for extracting continuous and discrete relaxation spectra
    from time-domain G(t) data.

    Parameters
    ----------
    config : ReSpectConfig, optional
        Solver configuration. Defaults to ReSpectConfig() if not supplied.

    Attributes
    ----------
    config : ReSpectConfig
        The active configuration.
    continuous : ContinuousResult or None
        Results from the continuous spectrum fit. None until fit() is called.
    discrete : DiscreteResult or None
        Results from the discrete spectrum fit. None until fit() is called.
    t : np.ndarray or None
        Time points used in the fit (after any resampling).
    Gt : np.ndarray or None
        G(t) data used in the fit.
    weights : np.ndarray or None
        Per-datapoint weights used in the fit.
    """

    def __init__(self, config: Optional[ReSpectConfig] = None) -> None:
        self.config:     ReSpectConfig              = config or ReSpectConfig()
        self.continuous: Optional[ContinuousResult] = None
        self.discrete:   Optional[DiscreteResult]   = None
        self.t:          Optional[np.ndarray]       = None
        self.Gt:         Optional[np.ndarray]       = None
        self.weights:    Optional[np.ndarray]       = None

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str) -> ReSpect:
        """Construct a ReSpect solver from a TOML configuration file."""
        return cls(ReSpectConfig.from_toml(path))

    @classmethod
    def from_yaml(cls, path: str) -> ReSpect:
        """Construct a ReSpect solver from a YAML configuration file."""
        return cls(ReSpectConfig.from_yaml(path))

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        source:   Union[str, os.PathLike, Sequence, np.ndarray],
        Gt:       Optional[np.ndarray] = None,
        weights:  Optional[np.ndarray] = None,
        resample: Optional[bool]       = None,
    ) -> ReSpect:
        """Load G(t) data and compute continuous then discrete spectra.

        Parameters
        ----------
        source : path, array, or sequence of arrays
            The experimental data, in any of these forms:

            - ``fit("Gt.dat")`` : path to a text file with 2 columns
              ``[t, G(t)]`` or 3 columns ``[t, G(t), weights]``.
            - ``fit(t, Gt)`` : 1-D arrays as separate arguments.
            - ``fit((t, Gt))`` or ``fit((t, Gt, weights))`` : a tuple of
              1-D arrays.
            - ``fit(data)`` : a 2-D array with the same columns as the file.

        Gt : np.ndarray, optional
            Relaxation modulus, when *source* is the array of time points.
        weights : np.ndarray, optional
            Per-datapoint weights, shape (n,). An alternative to supplying
            them as a third column or third tuple entry. Defaults to 1.
        resample : bool, optional
            Whether to resample the data onto a geometric grid of
            ``config.n_resample`` points. Defaults to ``config.resample``.
            Data supplied with weights are treated as pre-processed and
            are never resampled.

        Returns
        -------
        self — supports method chaining.

        Raises
        ------
        ReSpectError
            If the data cannot be read or are inconsistently shaped.
        """
        if Gt is not None:
            source = (source, Gt)

        self.t, self.Gt, self.weights = load_data(
            source,
            weights=weights,
            resample=self.config.resample if resample is None else resample,
            n_resample=self.config.n_resample,
            warn_if_not_resampled=resample is True,
        )

        self.continuous = fit_continuous(
            self.t, self.Gt, self.weights, self.config
        )

        self.discrete = fit_discrete(
            self.t, self.Gt, self.weights, self.continuous, self.config
        )

        return self

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    @_accept_legacy_save_order
    def save(
        self,
        which: Union[str, list[str]]   = "base",
        path:  Union[str, os.PathLike] = "./",
    ) -> ReSpect:
        """Write result files to *path*.

        Parameters
        ----------
        which : "base" or "full"
            ``"base"`` writes crs.dat, drs.dat, Gfit.dat.
            ``"full"`` additionally writes rho-eta.dat, logPlam.dat, aic.dat.
        path : str or path-like
            Output directory (created if it does not exist).

        Returns
        -------
        self — supports method chaining.
        """
        self._check_fitted("save")
        _save(
            which=which,
            path=path,
            t=self.t,
            cont_result=self.continuous,
            disc_result=self.discrete,
            plateau=self.config.plateau,
        )
        return self

    # ------------------------------------------------------------------
    # plot
    # ------------------------------------------------------------------

    def plot(
        self,
        which:  Union[str, list[str]]   = "base",
        toFile: bool                    = False,
        path:   Union[str, os.PathLike] = "./",
    ) -> list:
        """Plot spectra and diagnostics.

        Parameters
        ----------
        which : "base" or "full"
            ``"base"`` produces a two-panel figure: exp(H(s)) with error
            band and discrete modes (left), G(t) data vs fits (right).
            ``"full"`` additionally produces a three-panel diagnostic
            figure: log p(λ), ρ-η L-curve, AIC scan.
        toFile : bool
            If True, save figures as PDFs to *path* instead of
            displaying them.
        path : str or path-like
            Output directory for PDFs (used only when toFile=True).

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            The figures produced, base figure first.
        """
        self._check_fitted("plot")
        return _plot(
            which=which,
            toFile=toFile,
            path=path,
            t=self.t,
            Gt=self.Gt,
            cont_result=self.continuous,
            disc_result=self.discrete,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ReSpect("
            f"fitted={self.continuous is not None}, "
            f"ns={self.config.ns}, "
            f"plateau={self.config.plateau}, "
            f"freq_end='{self.config.freq_end}')"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self, caller: str) -> None:
        if self.continuous is None or self.discrete is None:
            raise ReSpectError(
                f"ReSpect.{caller}() called before fit(). "
                "Run solver.fit(source) first."
            )
