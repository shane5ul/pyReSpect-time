"""
io.py — Data loading and file output for pyReSpect.

Handles all file I/O for the pyReSpect library, cleanly separated
from the computation layer. No scientific computation happens here.

Functions
---------
load_data(source, weights, resample, n_resample)
    Load experimental data from a file or numpy arrays.

save(path, which, t, cont_result, disc_result)
    Write results to files in the specified output directory.

Private
-------
_resample_geometric(t, Gt, n)
    Resample data onto a geometric grid via linear interpolation.

_validate_which(which, cont_result, disc_result)
    Validate the 'which' argument and check results exist.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from .config import ReSpectError
from .continuous import ContinuousResult
from .discrete import DiscreteResult


# Valid 'which' tokens
_VALID_WHICH = {"base", "full"}

# Both tokens require both results
_NEEDS_CONT = {"base", "full"}
_NEEDS_DISC = {"base", "full"}


# ---------------------------------------------------------------------------
# Public: load_data
# ---------------------------------------------------------------------------

def load_data(
    source:     Union[str, tuple],
    weights:    Optional[np.ndarray] = None,
    resample:   bool                 = True,
    n_resample: int                  = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load experimental stress relaxation data.

    Accepts either a file path or a pair of numpy arrays. Optionally
    resamples the data onto a geometrically-spaced grid (opt-out via
    resample=False).

    Parameters
    ----------
    source : str or (np.ndarray, np.ndarray)
        Either:
        - A path to a data file with 2 columns [t, G(t)] or
          3 columns [t, G(t), weights].
        - A tuple (t, Gt) of numpy arrays.
    weights : np.ndarray or None, optional
        Per-datapoint weights. Only used when source is a tuple.
        Ignored when source is a 3-column file (weights are read
        from the file). If None and source is a 2-column file or
        tuple, weights default to 1.0.
    resample : bool, optional
        If True (default), resample 2-column file data or tuple data
        onto a geometric grid of n_resample points. Has no effect on
        3-column file data, which is assumed to be pre-processed.
    n_resample : int, optional
        Number of points for geometric resampling. Default: 100.
        Only used when resample=True.

    Returns
    -------
    t : np.ndarray, shape (n,)
        Time points.
    Gt : np.ndarray, shape (n,)
        Relaxation modulus G(t).
    w : np.ndarray, shape (n,)
        Per-datapoint weights.

    Raises
    ------
    ReSpectError
        If the file cannot be read, is incorrectly formatted, or the
        array shapes are inconsistent.
    """
    if isinstance(source, str):
        t, Gt, w, is_preprocessed = _load_from_file(source)
    elif isinstance(source, tuple) and len(source) == 2:
        t, Gt = source
        t     = np.asarray(t,  dtype=float)
        Gt    = np.asarray(Gt, dtype=float)

        if t.shape != Gt.shape or t.ndim != 1:
            raise ReSpectError(
                "When source is a tuple, t and Gt must be 1-D arrays "
                "of the same length."
            )

        w               = (np.asarray(weights, dtype=float)
                           if weights is not None
                           else np.ones(len(t)))
        is_preprocessed = False
    else:
        raise ReSpectError(
            "source must be a file path (str) or a tuple (t, Gt) of "
            "numpy arrays."
        )

    # Resample unless the data is pre-processed (3-column file)
    if resample and not is_preprocessed:
        t, Gt = _resample_geometric(t, Gt, n_resample)
        w     = np.ones(len(t))         # weights reset after resampling

    return t, Gt, w


# ---------------------------------------------------------------------------
# Public: save
# ---------------------------------------------------------------------------

def save(
    path:        str,
    which:       Union[str, list[str]],
    t:           np.ndarray,
    cont_result: Optional[ContinuousResult] = None,
    disc_result: Optional[DiscreteResult]   = None,
) -> None:
    """Write results to files in the specified output directory.

    Parameters
    ----------
    path : str
        Output directory. Created if it does not exist.
    which : str or list of str
        Which outputs to write. Valid values:

        - ``"base"`` : crs.dat (exp(H)), drs.dat (g, tau), and Gfit.dat 
                       Gfit.dat contains three columns: (t, Gt_crs, Gt_drs).
                       
        - ``"full"`` : above + rho-eta.dat, logPlam.dat, aic.dat.
                       Diagnostic files are silently skipped if the L-curve
                       was not computed (i.e. lam_C was pre-specified).

    t : np.ndarray, shape (n,)
        Experimental time points.
    cont_result : ContinuousResult or None
        Required for both tokens.
    disc_result : DiscreteResult or None
        Required for both tokens.

    Raises
    ------
    ReSpectError
        If a requested output requires a result that has not been computed,
        or if an invalid 'which' token is supplied.
    """
    tokens = _parse_which(which)
    _validate_which(tokens, cont_result, disc_result)
    os.makedirs(path, exist_ok=True)

    for token in tokens:

        if token == "base":
            _write_base(path, t, cont_result, disc_result)

        elif token == "full":
            _write_base(path, t, cont_result, disc_result)
            _write_full(path, cont_result, disc_result)


# ---------------------------------------------------------------------------
# Private: file loading
# ---------------------------------------------------------------------------

def _load_from_file(
    fname: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Load data from a 2- or 3-column text file.

    Returns
    -------
    t, Gt, w : np.ndarray
    is_preprocessed : bool
        True for 3-column files (weights supplied; no resampling needed).
    """
    try:
        data = np.loadtxt(fname)
    except OSError:
        raise ReSpectError(
            f"Could not read data file '{fname}'. "
            "Check that the path is correct and the file is properly formatted."
        )

    if data.ndim != 2 or data.shape[1] not in (2, 3):
        raise ReSpectError(
            f"Data file '{fname}' must have 2 columns [t, G(t)] "
            "or 3 columns [t, G(t), weights]."
        )

    t   = data[:, 0]
    Gt  = data[:, 1]

    # Remove duplicate time values
    t, idx = np.unique(t, return_index=True)
    Gt     = Gt[idx]

    if data.shape[1] == 3:
        w = data[:, 2][idx]
        return t, Gt, w, True       # pre-processed; skip resampling
    else:
        return t, Gt, np.ones(len(t)), False


def _resample_geometric(
    t:  np.ndarray,
    Gt: np.ndarray,
    n:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample (t, Gt) onto n geometrically-spaced time points.

    Uses linear interpolation. The resampled grid spans [t_min, t_max].

    Parameters
    ----------
    t : np.ndarray, shape (m,)
    Gt : np.ndarray, shape (m,)
    n : int
        Number of output points.

    Returns
    -------
    t_new : np.ndarray, shape (n,)
    Gt_new : np.ndarray, shape (n,)
    """
    from scipy.interpolate import interp1d

    f      = interp1d(t, Gt, fill_value='extrapolate')
    t_new  = np.geomspace(t.min(), t.max(), n)
    Gt_new = f(t_new)

    return t_new, Gt_new


# ---------------------------------------------------------------------------
# Private: write helpers
# ---------------------------------------------------------------------------

def _write_base(
    path:        str,
    t:           np.ndarray,
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> None:
    """Write crs.dat, drs.dat, and combined Gfit.dat.

    Gfit.dat columns: t, G_continuous, G_discrete.
    crs.dat and drs.dat include a G0 header line when plateau is active.
    """
    # crs.dat: [s, exp(H)] with optional G0 header
    h_path = os.path.join(path, "crs.dat")
    if cont_result.G0 is not None:
        np.savetxt(
            h_path,
            np.c_[cont_result.s, np.exp(cont_result.H)],
            fmt='%e',
            header=f'G0 = {cont_result.G0:.6e}',
        )
    else:
        np.savetxt(
            h_path,
            np.c_[cont_result.s, np.exp(cont_result.H)],
            fmt='%e',
        )

    # dmodes.dat: [g, tau, dtau] with optional G0 header
    dmodes_path = os.path.join(path, "drs.dat")
    if disc_result.G0 is not None:
        np.savetxt(
            dmodes_path,
            np.c_[disc_result.g, disc_result.tau, disc_result.dtau],
            fmt='%e',
            header=f'G0 = {disc_result.G0:.6e}',
        )
    else:
        np.savetxt(
            dmodes_path,
            np.c_[disc_result.g, disc_result.tau, disc_result.dtau],
            fmt='%e',
        )

    # Gfit.dat: [t, G_continuous, G_discrete]
    np.savetxt(
        os.path.join(path, "Gfit.dat"),
        np.c_[t, cont_result.G_fit, disc_result.G_fit],
        fmt='%e',
        header='t G_continuous G_discrete',
    )


def _write_full(
    path:        str,
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> None:
    """Write diagnostic files: rho-eta.dat, logPlam.dat, aic.dat.

    L-curve files (rho-eta.dat, logPlam.dat) are silently
    skipped when lam_C was pre-specified and the L-curve was not computed.
    """
    # L-curve diagnostics — only available when lam_C was auto-determined
    if cont_result.lam is not None:

        # rho-eta.dat: [lam, rho, eta]
        np.savetxt(
            os.path.join(path, "rho-eta.dat"),
            np.c_[cont_result.lam, cont_result.rho, cont_result.eta],
            fmt='%e',
        )

        # logPlam.dat: [lam, log_P]
        np.savetxt(
            os.path.join(path, "logPlam.dat"),
            np.c_[cont_result.lam, cont_result.log_P],
            fmt='%e',
        )

    # aic.dat: [wt_base, N_bst, AIC_bst]
    np.savetxt(
        os.path.join(path, "aic.dat"),
        np.c_[disc_result.wt_base, disc_result.N_bst, disc_result.AIC_bst],
        fmt='%f\t%i\t%e',
    )


# ---------------------------------------------------------------------------
# Private: validation helpers
# ---------------------------------------------------------------------------

def _parse_which(which: Union[str, list[str]]) -> list[str]:
    """Normalise 'which' to a list of strings and validate tokens."""
    if isinstance(which, str):
        tokens = [which]
    else:
        tokens = list(which)

    invalid = set(tokens) - _VALID_WHICH
    if invalid:
        raise ReSpectError(
            f"Invalid 'which' value(s): {invalid}. "
            f"Must be one or more of {_VALID_WHICH}."
        )

    return tokens


def _validate_which(
    tokens:      list[str],
    cont_result: Optional[ContinuousResult],
    disc_result: Optional[DiscreteResult],
) -> None:
    """Raise ReSpectError if a requested output's result is missing."""
    for token in tokens:
        if token in _NEEDS_CONT and cont_result is None:
            raise ReSpectError(
                f"'{token}' requires a continuous spectrum result, "
                "but none is available. Run fit() first."
            )
        if token in _NEEDS_DISC and disc_result is None:
            raise ReSpectError(
                f"'{token}' requires a discrete spectrum result, "
                "but none is available. Run fit() first."
            )
