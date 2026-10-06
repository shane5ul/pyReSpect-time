"""
discrete.py — Discrete relaxation spectrum solver for pyReSpect-time.

Given the continuous spectrum H(s) from fit_continuous(), extracts a
discrete set of Maxwell modes {g_i, τ_i} that best represent the
stress relaxation modulus G(t):

    G(t) = Σ_i g_i exp(-t/τ_i)  [+ G0]

The optimal number of modes N is determined by minimizing the AIC
criterion over a range of candidate values, scanning a grid of base
weight distributions.

Public API
----------
fit_discrete(t, Gt, weights, cont_result, config) -> DiscreteResult

All other functions are private to this module.

The structure of this module, the algorithm constants below, and the
DiscreteResult fields are deliberately the same as in pyrespect_freq;
only the kernel differs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from scipy.optimize import nnls, least_squares, minimize

from .config import ReSpectConfig, ReSpectWarning
from .continuous import ContinuousResult


# ---------------------------------------------------------------------------
# Algorithm constants (identical in pyrespect_time and pyrespect_freq)
# ---------------------------------------------------------------------------

#: Smallest number of modes considered in the AIC scan.
_N_MIN = 2

#: Modes with g_i / max(g) below this are dropped as negligible.
_PRUNE_TOL = 1e-7

#: Allowed range of τ relative to the data window, as (low, high):
#: low * t_min <= τ <= high * t_max. Keeps modes from running away to
#: where the data cannot constrain them.
_TAU_WINDOW = (0.02, 50.0)

#: Maximum number of passes of the merge-close-modes loop.
_MAX_MERGE_TRIES = 3


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiscreteResult:
    """Results from the discrete spectrum fit.

    The field names are the same in pyrespect_time and pyrespect_freq.

    Attributes
    ----------
    g : np.ndarray, shape (N,)
        Maxwell mode weights g_i.
    tau : np.ndarray, shape (N,)
        Maxwell relaxation times τ_i, in increasing order.
    dtau : np.ndarray, shape (N,)
        Uncertainty estimates for τ_i. Entries are np.nan when the NLLS
        fine-tuning step was not used.
    N : int
        Number of Maxwell modes returned (len(g)).
    G_fit : np.ndarray, shape (n,)
        Reconstructed G(t) from the discrete spectrum.
    G0 : float
        Plateau modulus. 0.0 if config.plateau is False.
    error : float
        Weighted sum of squared relative residuals of the discrete fit.
    wt_base : np.ndarray
        Scanned base weight values w_b.
    AIC_bst : np.ndarray
        Best AIC value at each w_b.
    N_bst : np.ndarray
        Number of modes requested (the N that enters the AIC penalty)
        at the best AIC for each w_b.
    nz_N_bst : np.ndarray
        Number of modes that survive pruning at the best AIC for each w_b.
    """
    g:        np.ndarray
    tau:      np.ndarray
    dtau:     np.ndarray
    N:        int
    G_fit:    np.ndarray
    G0:       float = 0.0
    error:    float = 0.0

    # AIC scan diagnostics, used by save/plot(which="full")
    wt_base:  Optional[np.ndarray] = None
    AIC_bst:  Optional[np.ndarray] = None
    N_bst:    Optional[np.ndarray] = None
    nz_N_bst: Optional[np.ndarray] = None

    # -- deprecated name ------------------------------------------------
    @property
    def N_opt(self) -> int:
        """Deprecated. Number of modes requested at the AIC optimum.

        This is not the number of modes returned; use ``N`` for that.
        """
        warnings.warn(
            "DiscreteResult.N_opt is deprecated: it is the number of modes "
            "requested at the AIC optimum, not the number returned. Use N "
            "(= len(g)) or N_bst[np.argmin(AIC_bst)].",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(self.N_bst[np.argmin(self.AIC_bst)])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_discrete(
    t:           np.ndarray,
    Gt:          np.ndarray,
    weights:     np.ndarray,
    cont_result: ContinuousResult,
    config:      ReSpectConfig,
) -> DiscreteResult:
    """Fit a discrete Maxwell spectrum to stress relaxation data.

    Uses the continuous spectrum from fit_continuous() to guide the
    placement of discrete modes, selects their number by minimizing

        AIC = 2 N + 2 C_error * error(N)

    over a grid of base weights w_b, fine-tunes the τ positions by NLLS,
    and merges modes that are too close (τ_{i+1}/τ_i < min_tau_spacing).

    Parameters
    ----------
    t : np.ndarray, shape (n,)
        Experimental time points.
    Gt : np.ndarray, shape (n,)
        Experimental relaxation modulus G(t).
    weights : np.ndarray, shape (n,)
        Per-datapoint weights.
    cont_result : ContinuousResult
        Output of fit_continuous(). Provides s, H, and G_fit.
    config : ReSpectConfig
        Configuration object.

    Returns
    -------
    DiscreteResult
    """
    s       = cont_result.s
    H       = cont_result.H
    plateau = config.plateau

    # ------------------------------------------------------------------
    # Range of N to scan
    # ------------------------------------------------------------------
    Nv   = _mode_counts(np.max(t) / np.min(t), len(t), config.max_num_modes)
    npts = len(Nv)

    # ------------------------------------------------------------------
    # Estimate error weight from continuous curve fit (AIC criterion)
    # ------------------------------------------------------------------
    Gc      = cont_result.G_fit
    C_error = 1.0 / np.std(weights * (Gc / Gt - 1.0))

    # ------------------------------------------------------------------
    # Scan base weight distributions
    # ------------------------------------------------------------------
    delta    = config.delta_base_weight_dist
    wt_base  = delta * np.arange(1, int(1.0 / delta))

    n_wb     = len(wt_base)
    AIC_bst  = np.zeros(n_wb)
    N_bst    = np.zeros(n_wb, dtype=int)
    nz_N_bst = np.zeros(n_wb, dtype=int)

    for ib, wb in enumerate(wt_base):

        wt    = _get_weights(H, t, s, wb)
        ev    = np.zeros(npts)
        nz_Nv = np.zeros(npts, dtype=int)

        for i, N in enumerate(Nv):
            z, _             = _grid_density(np.log(s), wt, N)
            _, tau, ev[i], _ = _maxwell_modes(z, t, Gt, weights, plateau)
            nz_Nv[i]         = len(tau)

        AIC          = 2.0 * Nv + 2.0 * C_error * ev
        AIC_bst[ib]  = np.min(AIC)
        N_bst[ib]    = Nv[np.argmin(AIC)]
        nz_N_bst[ib] = nz_Nv[np.argmin(AIC)]

    # ------------------------------------------------------------------
    # Global optimum: recompute, then fine-tune
    # ------------------------------------------------------------------
    i_best = np.argmin(AIC_bst)
    wt     = _get_weights(H, t, s, wt_base[i_best])
    z, _   = _grid_density(np.log(s), wt, int(N_bst[i_best]))

    g, tau, _, _ = _maxwell_modes(z, t, Gt, weights, plateau)
    dtau         = np.full(len(tau), np.nan)

    ok, g_f, tau_f, dtau_f = _fine_tune(tau, t, Gt, weights, plateau)
    if ok:
        g, tau, dtau = g_f, tau_f, dtau_f

    # ------------------------------------------------------------------
    # Merge modes that are too close
    # ------------------------------------------------------------------
    itry = 0
    while len(tau) > 1 and itry < _MAX_MERGE_TRIES:
        tau_spacing = tau[1:] / tau[:-1]
        if np.min(tau_spacing) >= config.min_tau_spacing:
            break

        tau_m = _merge_modes(g, tau, int(np.argmin(tau_spacing)))

        ok, g_f, tau_f, dtau_f = _fine_tune(tau_m, t, Gt, weights, plateau)
        if ok:
            g, tau, dtau = g_f, tau_f, dtau_f
        else:
            g, tau, _, _ = _maxwell_modes(np.log(tau_m), t, Gt, weights, plateau)
            dtau         = np.full(len(tau), np.nan)

        itry += 1

    # ------------------------------------------------------------------
    # Extract G0 and compute G_fit
    # ------------------------------------------------------------------
    G0 = 0.0
    if plateau:
        G0 = float(g[-1])
        g  = g[:-1]

    G_fit = _maxwell_kernel(tau, t) @ g + G0
    error = float(np.sum((weights * (G_fit / Gt - 1.0)) ** 2))

    return DiscreteResult(
        g=g,
        tau=tau,
        dtau=dtau,
        N=len(g),
        G_fit=G_fit,
        G0=G0,
        error=error,
        wt_base=wt_base,
        AIC_bst=AIC_bst,
        N_bst=N_bst,
        nz_N_bst=nz_N_bst,
    )


# ---------------------------------------------------------------------------
# Private: kernel (the only domain-specific part of this module)
# ---------------------------------------------------------------------------

def _maxwell_kernel(tau: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Maxwell kernel exp(-t_i/τ_j), shape (n, N)."""
    S, T = np.meshgrid(tau, t)
    return np.exp(-T / S)


def _design_matrix(tau: np.ndarray, t: np.ndarray, plateau: bool) -> np.ndarray:
    """Kernel with an extra column of ones for G0 when plateau=True."""
    K = _maxwell_kernel(tau, t)
    if plateau:
        K = np.hstack((K, np.ones((len(t), 1))))
    return K


def _model(t: np.ndarray, tau: np.ndarray, g: np.ndarray, plateau: bool) -> np.ndarray:
    """Model prediction for modes (g, tau); G0 is g[-1] when plateau=True."""
    if not plateau:
        return _maxwell_kernel(tau, t) @ g
    G = _maxwell_kernel(tau, t) @ g[:-1]
    G = G + g[-1]
    return G


def _tau_bounds(t: np.ndarray) -> tuple[float, float]:
    """Allowed range of τ for the given data window."""
    return _TAU_WINDOW[0] * np.min(t), _TAU_WINDOW[1] * np.max(t)


# ---------------------------------------------------------------------------
# Private: algorithm
# ---------------------------------------------------------------------------

def _mode_counts(
    span:          float,
    n:             int,
    max_num_modes: Optional[int],
) -> np.ndarray:
    """Values of N scanned by the AIC search.

    Parameters
    ----------
    span : float
        Ratio of the largest to the smallest abscissa of the data.
    n : int
        Number of data abscissae.
    max_num_modes : int or None
        User cap on N.

    Returns
    -------
    Nv : np.ndarray of int
    """
    decades = np.log10(span)
    N_min   = int(max(np.floor(0.5 * decades), _N_MIN))
    N_max   = int(min(np.floor(3.0 * decades), n / 4))

    if max_num_modes is not None:
        N_max = min(N_max, max_num_modes)

    if N_max > N_min:
        return np.arange(N_min, N_max + 1, dtype=int)

    warnings.warn(
        f"Only N = {N_max} is scanned for the discrete spectrum; "
        "make sure max_num_modes is set prudently.",
        ReSpectWarning,
        stacklevel=4,
    )
    return np.arange(N_max, N_max + 1, dtype=int)


def _nn_lls(
    t:        np.ndarray,
    tau:      np.ndarray,
    Gexp:     np.ndarray,
    wexp:     np.ndarray,
    plateau:  bool,
) -> tuple[np.ndarray, float]:
    """Solve the non-negative least squares problem for Maxwell weights.

    Minimizes || w * (K g / G_exp - 1) ||^2 subject to g >= 0.

    Parameters
    ----------
    t : np.ndarray, shape (n,)
    tau : np.ndarray, shape (N,)
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    plateau : bool
        If True, G0 is fitted as an extra, last element of g.

    Returns
    -------
    g : np.ndarray
        Non-negative weights (and G0 as last element if plateau=True).
    error : float
        Weighted residual sum of squares.
    """
    K  = _design_matrix(tau, t, plateau)

    # Weight the system: minimizes w*(Kg/Gexp - 1)^2
    Kp = (wexp / Gexp).reshape(-1, 1) * K
    g  = nnls(Kp, wexp, maxiter=100000)[0]

    error = float(np.sum((wexp * (K @ g / Gexp - 1.0)) ** 2))

    return g, error


def _maxwell_modes(
    z:       np.ndarray,
    t:       np.ndarray,
    Gexp:    np.ndarray,
    wexp:    np.ndarray,
    plateau: bool,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Solve for Maxwell modes at log-spaced positions z = log(τ).

    Solves NNLS, then drops modes outside the allowed τ window and modes
    with negligible weight (g_i / max(g) < _PRUNE_TOL), and sorts the
    rest by τ.

    Parameters
    ----------
    z : np.ndarray, shape (N,)
        Log relaxation times log(τ).
    t : np.ndarray, shape (n,)
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    plateau : bool

    Returns
    -------
    g : np.ndarray
        Weights of the surviving modes (G0 appended if plateau=True).
    tau : np.ndarray
        Relaxation times of the surviving modes, increasing.
    error : float
        Weighted residual sum of squares of the NNLS solution.
    keep : np.ndarray of int
        Indices into z of the surviving modes, in the order returned.
    """
    tau      = np.exp(z)
    g, error = _nn_lls(t, tau, Gexp, wexp, plateau)

    g_modes  = g[:-1] if plateau else g
    lo, hi   = _tau_bounds(t)

    in_window = (tau >= lo) & (tau <= hi)
    keep      = np.where(in_window)[0]
    g_ref     = np.max(g_modes[keep]) if len(keep) else 1.0
    keep      = keep[g_modes[keep] / g_ref >= _PRUNE_TOL]
    keep      = keep[np.argsort(tau[keep])]

    g_out = g_modes[keep]
    if plateau:
        g_out = np.append(g_out, g[-1])

    return g_out, tau[keep], error, keep


def _get_weights(
    H:  np.ndarray,
    t:  np.ndarray,
    s:  np.ndarray,
    wb: float,
) -> np.ndarray:
    """Compute the weight of each relaxation mode on the continuous axis.

    Weights reflect each mode's average contribution to G(t), blended
    with a uniform baseline controlled by wb.

    Parameters
    ----------
    H : np.ndarray, shape (ns,)
        Log continuous spectrum.
    t : np.ndarray, shape (n,)
    s : np.ndarray, shape (ns,)
    wb : float
        Base weight blending factor in [0, 1).

    Returns
    -------
    wt : np.ndarray, shape (ns,)
        Normalized weights for mode placement.
    """
    ns = len(s)

    # Trapezoidal weights in log-space
    hs        = np.zeros(ns)
    hs[0]     = 0.5 * np.log(s[1] / s[0])
    hs[-1]    = 0.5 * np.log(s[-1] / s[-2])
    hs[1:-1]  = 0.5 * (np.log(s[2:]) - np.log(s[:-2]))

    kern = _maxwell_kernel(s, t)                  # (n, ns)

    # Contribution of each (t_i, s_j) pair, weighted by H
    wij = kern * (hs * np.exp(H)).reshape(1, ns)  # (n, ns)
    K   = wij.sum(axis=1)                         # (n,)  = G(t_i)

    # Normalize rows so each row sums to 1
    wij = wij / K.reshape(-1, 1)                  # (n, ns)

    # Sum contributions across all time points
    wt  = wij.sum(axis=0)                         # (ns,)
    wt  = wt / np.trapezoid(wt, np.log(s))

    # Blend with uniform baseline
    wt  = (1.0 - wb) * wt + wb * np.mean(wt) * np.ones(ns)

    return wt


def _grid_density(
    x:  np.ndarray,
    px: np.ndarray,
    N:  int,
) -> tuple[np.ndarray, np.ndarray]:
    """Distribute N points according to a density function px(x).

    Places quadrature points such that each interval carries equal
    probability mass under px, with endpoints of x always included.

    Parameters
    ----------
    x : np.ndarray
        Domain points (need not be equispaced).
    px : np.ndarray
        Density or probability distribution (positive, need not be normalized).
    N : int
        Number of output points.

    Returns
    -------
    z : np.ndarray, shape (N,)
        Points distributed according to px.
    h : np.ndarray, shape (N,)
        Interval widths (useful for quadrature).
    """
    npts = 100
    xi   = np.linspace(x.min(), x.max(), npts)
    fint = interp1d(x, px, kind='cubic')
    pint = fint(xi)

    ci   = cumulative_trapezoid(pint, xi, initial=0)
    pint = pint / ci[-1]
    ci   = ci   / ci[-1]

    alfa    = 1.0 / max((N - 1), 1)
    zij     = np.zeros(N + 1)
    z       = np.zeros(N)
    z[0]    = x.min()
    z[-1]   = x.max()

    beta       = np.arange(0.5, N - 0.5) * alfa
    zij[0]     = z[0]
    zij[-1]    = z[-1]
    fint_inv   = interp1d(ci, xi, kind='cubic')
    zij[1:N]   = fint_inv(beta)
    h          = np.diff(zij)

    beta     = np.arange(1, N - 1) * alfa
    z[1:-1]  = fint_inv(beta)

    return z, h


def _merge_modes(
    g:     np.ndarray,
    tau:   np.ndarray,
    imode: int,
) -> np.ndarray:
    """Merge modes imode and imode+1 into a single mode.

    Finds the best single-mode approximation to the two-mode pair
    by minimizing the integrated squared relative difference.

    Parameters
    ----------
    g : np.ndarray
        Current mode weights.
    tau : np.ndarray
        Current relaxation times.
    imode : int
        Index of the first mode to merge.

    Returns
    -------
    tau_new : np.ndarray
        Updated relaxation times with one fewer mode.
    """
    g1, tau1 = g[imode],     tau[imode]
    g2, tau2 = g[imode + 1], tau[imode + 1]

    def _integrand(t: float, gn: float, taun: float) -> float:
        Gn = gn * np.exp(-t / taun)
        Go = g1 * np.exp(-t / tau1) + g2 * np.exp(-t / tau2)
        return (Gn / Go - 1.0) ** 2

    def _cost(par: np.ndarray) -> float:
        """Integrated squared relative error between merged and original."""
        tmin = min(tau1, tau2) / 10.0
        tmax = max(tau1, tau2) * 10.0
        return quad(_integrand, tmin, tmax, args=(par[0], par[1]))[0]

    res = minimize(_cost, np.array([g1 + g2, 0.5 * (tau1 + tau2)]))

    tau_new        = np.delete(tau, imode + 1)
    tau_new[imode] = res.x[1]

    return tau_new


def _fine_tune(
    tau:     np.ndarray,
    t:       np.ndarray,
    Gexp:    np.ndarray,
    wexp:    np.ndarray,
    plateau: bool,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Fine-tune mode positions via non-linear least squares.

    Refines the τ positions by NLLS (weights g are re-solved by NNLS at
    every step), keeping τ inside the allowed window. The refinement is
    reported as successful only if it converges and does not make the
    fit worse than the starting point.

    Parameters
    ----------
    tau : np.ndarray, shape (N,)
        Initial relaxation times.
    t : np.ndarray, shape (n,)
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    plateau : bool

    Returns
    -------
    success : bool
        If False, the caller should keep its starting solution.
    g : np.ndarray
        Weights at the refined positions (G0 appended if plateau=True).
    tau : np.ndarray
        Refined relaxation times, increasing.
    dtau : np.ndarray
        Uncertainty estimates for tau from the NLLS Jacobian.
    """

    def _residuals(tau_: np.ndarray) -> np.ndarray:
        g_, _ = _nn_lls(t, tau_, Gexp, wexp, plateau)
        return wexp * (_model(t, tau_, g_, plateau) / Gexp - 1.0)

    init_error = np.linalg.norm(_residuals(tau))

    try:
        res  = least_squares(_residuals, tau, bounds=_tau_bounds(t))
        cov  = np.linalg.pinv(res.jac.T @ res.jac) * (res.fun ** 2).mean()
        dtau = np.sqrt(np.diag(cov))
    except Exception:
        warnings.warn(
            "NLLS fine-tuning of the discrete modes did not converge; "
            "keeping the NNLS solution. Uncertainty estimates (dtau) "
            "will be NaN.",
            ReSpectWarning,
            stacklevel=2,
        )
        g, tau, _, _ = _maxwell_modes(np.log(tau), t, Gexp, wexp, plateau)
        return False, g, tau, np.full(len(tau), np.nan)

    # Re-solve NNLS at the refined positions; modes may drop out
    g, tau, _, keep = _maxwell_modes(np.log(res.x), t, Gexp, wexp, plateau)
    final_error     = np.linalg.norm(_residuals(tau))

    return bool(final_error <= init_error), g, tau, dtau[keep]
