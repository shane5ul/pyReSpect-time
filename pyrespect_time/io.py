"""
io.py — Data loading and file output for pyReSpect-time.

All file I/O is confined here. No scientific computation happens in
this module.

Functions
---------
load_data(source, weights, resample, n_resample)
    Load G(t) data from a file, a tuple of arrays, or a 2-D array.

save(which, path, t, cont_result, disc_result, plateau)
    Write results to files in the specified output directory.

The layout of this module, the names of the output files and their
headers are deliberately the same as in pyrespect_freq.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence, Union

import numpy as np

from .config import ReSpectError, ReSpectWarning
from .continuous import ContinuousResult
from .discrete import DiscreteResult


# Valid 'which' tokens
_VALID_WHICH = ("base", "full")

# Number of columns in the data: without / with per-datapoint weights
_NCOLS_RAW      = 2
_NCOLS_WEIGHTED = 3
_COLUMNS_DOC    = "2 columns [t, G(t)] or 3 columns [t, G(t), weights]"


# ---------------------------------------------------------------------------
# Public: load_data
# ---------------------------------------------------------------------------

def load_data(
    source:           Union[str, os.PathLike, Sequence, np.ndarray],
    weights:          Optional[np.ndarray] = None,
    resample:         bool                 = True,
    n_resample:       int                  = 100,
    warn_if_not_resampled: bool            = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load experimental stress relaxation data.

    Parameters
    ----------
    source : path, sequence of arrays, or 2-D array
        One of:

        - A path to a text file with 2 columns ``[t, G(t)]`` or
          3 columns ``[t, G(t), weights]``.
        - A sequence ``(t, Gt)`` or ``(t, Gt, weights)`` of 1-D arrays.
        - A 2-D array with the same column layout as the file.

    weights : np.ndarray or None, optional
        Per-datapoint weights, shape (n,). An alternative to supplying
        them as the third column / third array of *source*.
    resample : bool, optional
        If True (default), resample the data onto a geometric grid of
        *n_resample* points. Data that come with weights are treated as
        pre-processed and are never resampled.
    n_resample : int, optional
        Number of points for geometric resampling. Default: 100.
    warn_if_not_resampled : bool, optional
        Emit a ReSpectWarning when *resample* is True but resampling is
        skipped because weights were supplied.

    Returns
    -------
    t : np.ndarray, shape (n,)
        Time points, increasing, without duplicates.
    Gt : np.ndarray, shape (n,)
        Relaxation modulus G(t).
    w : np.ndarray, shape (n,)
        Per-datapoint weights.

    Raises
    ------
    ReSpectError
        If the source cannot be read or is incorrectly formatted.
    """
    cols = _get_columns(source)

    t, Gt = cols[0], cols[1]
    wt    = cols[2] if len(cols) == _NCOLS_WEIGHTED else None

    if weights is not None:
        if wt is not None:
            raise ReSpectError(
                "Weights were supplied twice: once with the data and once "
                "through the 'weights' argument."
            )
        wt = np.asarray(weights, dtype=float)
        if wt.shape != t.shape:
            raise ReSpectError(
                f"weights must have the same shape as t {t.shape}; "
                f"got {wt.shape}."
            )

    # Sort by t and remove duplicate time values
    t, idx = np.unique(t, return_index=True)
    Gt     = Gt[idx]
    if wt is not None:
        wt = wt[idx]

    # Data with weights are treated as pre-processed: never resampled
    if wt is None:
        if resample:
            t, Gt = _resample_geometric(t, Gt, n_resample)
        wt = np.ones(len(t))
    elif resample and warn_if_not_resampled:
        warnings.warn(
            "resample=True was ignored because weights were supplied; "
            "data with weights are used as-is.",
            ReSpectWarning,
            stacklevel=3,
        )

    return t, Gt, wt


# ---------------------------------------------------------------------------
# Public: save
# ---------------------------------------------------------------------------

def save(
    which:       Union[str, list[str]],
    path:        Union[str, os.PathLike],
    t:           np.ndarray,
    cont_result: Optional[ContinuousResult] = None,
    disc_result: Optional[DiscreteResult]   = None,
    plateau:     bool                       = False,
) -> None:
    """Write results to files in the specified output directory.

    Parameters
    ----------
    which : str or list of str
        Which outputs to write. Valid values:

        - ``"base"`` : crs.dat, drs.dat, Gfit.dat.
        - ``"full"`` : above + rho-eta.dat, logPlam.dat, aic.dat.
          The two L-curve files are skipped when lam_C was pre-specified.

    path : str or path-like
        Output directory. Created if it does not exist.
    t : np.ndarray, shape (n,)
        Experimental time points (needed to write Gfit.dat).
    cont_result : ContinuousResult or None
    disc_result : DiscreteResult or None
    plateau : bool
        If True, the fitted G0 is recorded in the headers of crs.dat
        and drs.dat.

    Raises
    ------
    ReSpectError
        If a requested output requires a result that has not been
        computed, or an invalid 'which' token is supplied.
    """
    tokens = _parse_which(which)
    _validate_which(tokens, cont_result, disc_result)
    os.makedirs(path, exist_ok=True)

    _write_base(path, t, cont_result, disc_result, plateau)
    if "full" in tokens:
        _write_full(path, cont_result, disc_result)


# ---------------------------------------------------------------------------
# Private: loading helpers
# ---------------------------------------------------------------------------

def _get_columns(source) -> list[np.ndarray]:
    """Turn any accepted *source* into a list of 1-D float columns."""
    if isinstance(source, (str, os.PathLike)):
        fname = os.fspath(source)
        try:
            data = np.loadtxt(fname)
        except (OSError, ValueError):
            raise ReSpectError(
                f"Could not read data file '{fname}'. "
                "Check that the path is correct and the file is properly "
                "formatted."
            ) from None
        where = f"Data file '{fname}'"

    elif isinstance(source, np.ndarray):
        data  = np.asarray(source, dtype=float)
        where = "A data array"

    elif isinstance(source, (tuple, list)):
        cols = [np.asarray(c, dtype=float) for c in source]
        if len(cols) not in (_NCOLS_RAW, _NCOLS_WEIGHTED):
            raise ReSpectError(
                f"A tuple source must have length {_NCOLS_RAW} (t, Gt) or "
                f"{_NCOLS_WEIGHTED} (t, Gt, weights); got {len(cols)}."
            )
        _check_shapes(*cols)
        return cols

    else:
        raise ReSpectError(
            "source must be a file path, a tuple of 1-D arrays, or a "
            f"2-D array with {_COLUMNS_DOC}."
        )

    if data.ndim != 2 or data.shape[1] not in (_NCOLS_RAW, _NCOLS_WEIGHTED):
        raise ReSpectError(f"{where} must have {_COLUMNS_DOC}.")

    return [data[:, j] for j in range(data.shape[1])]


def _check_shapes(*arrays: np.ndarray) -> None:
    """Raise ReSpectError if arrays are not all 1-D and the same length."""
    shapes = [a.shape for a in arrays]
    if any(a.ndim != 1 for a in arrays):
        raise ReSpectError(
            f"All input arrays must be 1-D; got shapes {shapes}."
        )
    if len(set(shapes)) > 1:
        raise ReSpectError(
            f"All input arrays must have the same length; "
            f"got shapes {shapes}."
        )


def _resample_geometric(
    t:  np.ndarray,
    Gt: np.ndarray,
    n:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample (t, Gt) onto n geometrically-spaced time points.

    Uses linear interpolation. The resampled grid spans [t_min, t_max].
    """
    from scipy.interpolate import interp1d

    f      = interp1d(t, Gt, fill_value='extrapolate')
    t_new  = np.geomspace(t.min(), t.max(), n)

    return t_new, f(t_new)


# ---------------------------------------------------------------------------
# Private: write helpers
# ---------------------------------------------------------------------------

def _header(columns: str, G0: Optional[float] = None) -> str:
    """File header: optional 'G0 = ...' line, then the column names."""
    if G0 is None:
        return columns
    return f"G0 = {G0:.6e}\n{columns}"


def _write_base(
    path:        Union[str, os.PathLike],
    t:           np.ndarray,
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
    plateau:     bool,
) -> None:
    """Write crs.dat, drs.dat, and Gfit.dat."""

    # crs.dat: [s, exp(H(s))]
    np.savetxt(
        os.path.join(path, "crs.dat"),
        np.c_[cont_result.s, np.exp(cont_result.H)],
        fmt="%e",
        header=_header("s  h", cont_result.G0 if plateau else None),
    )

    # drs.dat: [g_i, tau_i, dtau_i]
    np.savetxt(
        os.path.join(path, "drs.dat"),
        np.c_[disc_result.g, disc_result.tau, disc_result.dtau],
        fmt="%e",
        header=_header("g  tau  dtau", disc_result.G0 if plateau else None),
    )

    # Gfit.dat: [t, G_cont, G_disc]
    np.savetxt(
        os.path.join(path, "Gfit.dat"),
        np.c_[t, cont_result.G_fit, disc_result.G_fit],
        fmt="%e",
        header="t  G_cont  G_disc",
    )


def _write_full(
    path:        Union[str, os.PathLike],
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> None:
    """Write rho-eta.dat, logPlam.dat, aic.dat."""

    # L-curve files only available when lam_C was auto-determined
    if cont_result.lam is not None:
        np.savetxt(
            os.path.join(path, "rho-eta.dat"),
            np.c_[cont_result.lam, cont_result.rho, cont_result.eta],
            fmt="%e",
            header="lambda  rho  eta",
        )
        np.savetxt(
            os.path.join(path, "logPlam.dat"),
            np.c_[cont_result.lam, cont_result.log_P],
            fmt="%e",
            header="lambda  logP",
        )

    # AIC scan is always available
    np.savetxt(
        os.path.join(path, "aic.dat"),
        np.c_[disc_result.wt_base, disc_result.N_bst,
              disc_result.AIC_bst, disc_result.nz_N_bst],
        fmt="%f\t%i\t%e\t%i",
        header="wt_base  N_bst  AIC  nz_N_bst",
    )


# ---------------------------------------------------------------------------
# Private: validation helpers
# ---------------------------------------------------------------------------

def _parse_which(which: Union[str, list[str]]) -> list[str]:
    """Normalise 'which' to a list of strings and validate tokens."""
    tokens  = [which] if isinstance(which, str) else list(which)
    invalid = [tok for tok in tokens if tok not in _VALID_WHICH]
    if invalid:
        raise ReSpectError(
            f"Invalid 'which' value(s): {invalid}. "
            f"Must be one of {list(_VALID_WHICH)}."
        )
    return tokens


def _validate_which(
    tokens:      list[str],
    cont_result: Optional[ContinuousResult],
    disc_result: Optional[DiscreteResult],
) -> None:
    """Raise ReSpectError if a requested output's result is missing."""
    if tokens and (cont_result is None or disc_result is None):
        raise ReSpectError(
            f"'{tokens[0]}' requires fitted spectra, but none are "
            "available. Run fit() first."
        )
