"""
discrete.py — Discrete relaxation spectrum solver for pyReSpect.

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
from .kernels import get_kern_mat



# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiscreteResult:
    """Results from the discrete spectrum fit.

    Attributes
    ----------
    g : np.ndarray, shape (N,)
        Maxwell mode weights g_i.
    tau : np.ndarray, shape (N,)
        Maxwell relaxation times τ_i.
    dtau : np.ndarray, shape (N,)
        Uncertainty estimates for τ_i. Entries are np.nan where the
        NLLS fine-tuning step failed to converge.
    N_opt : int
        Optimal number of Maxwell modes.
    G_fit : np.ndarray, shape (n,)
        Reconstructed G(t) from the discrete spectrum.
    G0 : float or None
        Plateau modulus. None if config.plateau is False.
    wt_base : np.ndarray
        Scanned base weight values w_b.
    AIC_bst : np.ndarray
        Best AIC value at each w_b.
    N_bst : np.ndarray
        Best N (nominal) at each w_b.
    """
    g:       np.ndarray
    tau:     np.ndarray
    dtau:    np.ndarray
    N_opt:   int
    G_fit:   np.ndarray
    G0:      Optional[float]
    wt_base: np.ndarray
    AIC_bst: np.ndarray
    N_bst:   np.ndarray


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
    placement of discrete modes, then optimizes their number and
    positions via AIC minimization and NLLS fine-tuning.

    Parameters
    ----------
    t : np.ndarray, shape (n,)
        Experimental time points.
    Gt : np.ndarray, shape (n,)
        Experimental relaxation modulus G(t).
    weights : np.ndarray, shape (n,)
        Per-datapoint weights.
    cont_result : ContinuousResult
        Output of fit_continuous(). Provides s, H, and optionally G0.
    config : ReSpectConfig
        Configuration object.

    Returns
    -------
    DiscreteResult
    """
    s = cont_result.s
    H = cont_result.H
    n = len(t)

    # ------------------------------------------------------------------
    # Determine range of N to scan
    # ------------------------------------------------------------------
    log_t_range = np.log10(t[-1] / t[0])
    N_min = int(max(np.floor(0.5 * log_t_range), 2))
    N_max = int(min(np.floor(3.0 * log_t_range), n / 4))

    if config.max_num_modes is not None:
        N_max = min(N_max, config.max_num_modes)

    Nv   = np.arange(N_min, N_max + 1, dtype=int)
    npts = len(Nv)

    # ------------------------------------------------------------------
    # Estimate error weight from continuous curve fit (AIC criterion)
    # ------------------------------------------------------------------
    kern_mat = get_kern_mat(s, t)
    Gc       = cont_result.G_fit
    C_error  = 1.0 / np.std(weights * (Gc / Gt - 1.0))

    # ------------------------------------------------------------------
    # Scan base weight distributions
    # ------------------------------------------------------------------
    delta = config.delta_base_weight_dist
    wt_base = delta * np.arange(1, int(1.0 / delta))

    n_wb   = len(wt_base)
    AIC_bst = np.zeros(n_wb)
    N_bst   = np.zeros(n_wb, dtype=int)
    nz_N_bst = np.zeros(n_wb, dtype=int)

    for ib, wb in enumerate(wt_base):

        wt = _get_weights(H, t, s, wb)

        ev   = np.zeros(npts)
        nz_Nv = np.zeros(npts, dtype=int)

        for i, N in enumerate(Nv):
            z, _          = _grid_density(np.log(s), wt, N)
            g, tau, ev[i], _ = _maxwell_modes(z, t, Gt, weights, config.plateau)
            nz_Nv[i]      = len(g)

        AIC            = 2.0 * Nv + 2.0 * C_error * ev
        AIC_bst[ib]    = np.min(AIC)
        N_bst[ib]      = Nv[np.argmin(AIC)]
        nz_N_bst[ib]   = nz_Nv[np.argmin(AIC)]

    # ------------------------------------------------------------------
    # Global optimum
    # ------------------------------------------------------------------
    i_best = np.argmin(AIC_bst)
    N_opt  = int(N_bst[i_best])
    wb_opt = wt_base[i_best]

    # ------------------------------------------------------------------
    # Recompute at optimum and fine-tune
    # ------------------------------------------------------------------
    wt          = _get_weights(H, t, s, wb_opt)
    z, _        = _grid_density(np.log(s), wt, N_opt)
    g, tau, _, _ = _maxwell_modes(z, t, Gt, weights, config.plateau)
    g, tau, dtau = _fine_tune(tau, t, Gt, weights, config.plateau,
                               estimate_error=True)

    # ------------------------------------------------------------------
    # Merge modes that are too close
    # ------------------------------------------------------------------
    if len(tau) > 1:
        indx         = np.argsort(tau)
        tau          = tau[indx]
        tau_spacing  = tau[1:] / tau[:-1]
        itry         = 0

        if config.plateau:
            g[:-1] = g[indx]
        else:
            g = g[indx]

        while np.min(tau_spacing) < config.min_tau_spacing and itry < 3:
            imode        = np.argmin(tau_spacing)
            tau          = _merge_modes(g, tau, imode)
            g, tau, dtau = _fine_tune(tau, t, Gt, weights, config.plateau,
                                       estimate_error=True)
            if len(tau) > 1:
                tau_spacing = tau[1:] / tau[:-1]
            else:
                break
            itry += 1

    # ------------------------------------------------------------------
    # Extract G0 and compute G_fit
    # ------------------------------------------------------------------
    G0 = None
    if config.plateau:
        G0 = float(g[-1])
        g  = g[:-1]

    S, T  = np.meshgrid(tau, t)
    K     = np.exp(-T / S)
    G_fit = K @ g
    if G0 is not None:
        G_fit = G_fit + G0

    return DiscreteResult(
        g=g,
        tau=tau,
        dtau=dtau,
        N_opt=N_opt,
        G_fit=G_fit,
        G0=G0,
        wt_base=wt_base,
        AIC_bst=AIC_bst,
        N_bst=N_bst,
    )


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _nn_lls(
    t:        np.ndarray,
    tau:      np.ndarray,
    Gexp:     np.ndarray,
    wexp:     np.ndarray,
    plateau:  bool,
) -> tuple[np.ndarray, float, float]:
    """Solve the non-negative least squares problem for Maxwell weights.

    Minimizes || w * (K g / G_exp - 1) ||^2 subject to g >= 0.

    Parameters
    ----------
    t : np.ndarray, shape (n,)
    tau : np.ndarray, shape (N,)
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    plateau : bool
        If True, appends a column of ones to K to fit G0.

    Returns
    -------
    g : np.ndarray
        Non-negative weights (and G0 as last element if plateau=True).
    error : float
        Weighted residual sum of squares.
    cond_Kp : float
        Condition number of the weighted kernel matrix.
    """
    S, T = np.meshgrid(tau, t)
    K    = np.exp(-T / S)                        # (n, N)

    if plateau:
        K = np.hstack((K, np.ones((len(Gexp), 1))))

    # Weight the system: minimizes w*(Kg/Gexp - 1)^2
    Kp      = (wexp / Gexp).reshape(-1, 1) * K
    cond_Kp = np.linalg.cond(Kp)
    g       = nnls(Kp, wexp, maxiter=100000)[0]

    G_model = K @ g
    error   = float(np.sum((wexp * (G_model / Gexp - 1.0)) ** 2))

    return g, error, cond_Kp


def _maxwell_modes(
    z:       np.ndarray,
    t:       np.ndarray,
    Gexp:    np.ndarray,
    wexp:    np.ndarray,
    plateau: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve for Maxwell modes at log-spaced positions z = log(τ).

    Solves NNLS and prunes negligibly small modes (g_i / max(g) < 1e-7).

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
    tau : np.ndarray
    error : float
    cond_Kp : float
    """
    tau              = np.exp(z)
    g, error, cond_Kp = _nn_lls(t, tau, Gexp, wexp, plateau)

    # Prune negligibly small modes
    g_body = g[:-1] if plateau else g
    i_zero = np.where(g_body / np.max(g_body) < 1e-7)[0]

    tau = np.delete(tau, i_zero)
    g   = np.delete(g,   i_zero)

    return g, tau, error, cond_Kp


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
    n  = len(t)

    # Trapezoidal weights in log-space
    hs        = np.zeros(ns)
    hs[0]     = 0.5 * np.log(s[1] / s[0])
    hs[-1]    = 0.5 * np.log(s[-1] / s[-2])
    hs[1:-1]  = 0.5 * (np.log(s[2:]) - np.log(s[:-2]))

    S, T = np.meshgrid(s, t)
    kern = np.exp(-T / S)                        # (n, ns)

    # Contribution of each (t_i, s_j) pair, weighted by H
    wij = kern * (hs * np.exp(H)).reshape(1, ns) # (n, ns)
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
        Number of output points (>= 3).

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

    alfa    = 1.0 / (N - 1)
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

    def _cost(par: np.ndarray) -> float:
        """Integrated squared relative error between merged and original."""
        gn, taun = par[0], par[1]
        g1, tau1 = g[imode],     tau[imode]
        g2, tau2 = g[imode + 1], tau[imode + 1]

        tmin = min(tau1, tau2) / 10.0
        tmax = max(tau1, tau2) * 10.0

        def _integrand(t: float) -> float:
            Gn = gn * np.exp(-t / taun)
            Go = g1 * np.exp(-t / tau1) + g2 * np.exp(-t / tau2)
            return (Gn / Go - 1.0) ** 2

        return quad(_integrand, tmin, tmax)[0]

    ini_guess = np.array([
        g[imode] + g[imode + 1],
        0.5 * (tau[imode] + tau[imode + 1]),
    ])
    res = minimize(_cost, ini_guess)

    tau_new         = np.delete(tau, imode + 1)
    tau_new[imode]  = res.x[1]

    return tau_new


def _fine_tune(
    tau:            np.ndarray,
    t:              np.ndarray,
    Gexp:           np.ndarray,
    wexp:           np.ndarray,
    plateau:        bool,
    estimate_error: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine-tune mode positions via non-linear least squares.

    Attempts NLLS optimization of τ positions. If it fails, returns
    the input τ unchanged and fills dtau with np.nan.

    Parameters
    ----------
    tau : np.ndarray, shape (N,)
        Initial relaxation times.
    t : np.ndarray, shape (n,)
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    plateau : bool
    estimate_error : bool
        If True, estimate uncertainties dtau from the NLLS Jacobian.

    Returns
    -------
    g : np.ndarray
    tau : np.ndarray
    dtau : np.ndarray
        Uncertainty estimates; np.nan entries where estimation failed.
    """

    def _residuals(
        tau_: np.ndarray,
        texp: np.ndarray,
        Gexp_: np.ndarray,
        wexp_: np.ndarray,
        plateau_: bool,
    ) -> np.ndarray:
        g_, _, _ = _nn_lls(texp, tau_, Gexp_, wexp_, plateau_)
        S, T     = np.meshgrid(tau_, texp)
        G_model  = np.exp(-T / S) @ (g_[:-1] if plateau_ else g_)
        if plateau_:
            G_model = G_model + g_[-1]
        return wexp_ * (G_model / Gexp_ - 1.0)

    nlls_success = False
    dtau         = np.full(len(tau), np.nan)

    try:
        res  = least_squares(
            _residuals, tau,
            bounds=(0.0, np.inf),
            args=(t, Gexp, wexp, plateau),
        )
        tau  = res.x
        tau0 = tau.copy()

        if estimate_error:
            J    = res.jac
            cov  = np.linalg.pinv(J.T @ J) * (res.fun ** 2).mean()
            dtau = np.sqrt(np.diag(cov))

        nlls_success = True

    except Exception:
        warnings.warn(
            "NLLS fine-tuning did not converge; returning NNLS solution. "
            "Uncertainty estimates (dtau) will be NaN in the output.",
            ReSpectWarning,
            stacklevel=2,
        )

    # Re-solve NNLS at the (possibly updated) tau positions
    g, tau, _, _ = _maxwell_modes(np.log(tau), t, Gexp, wexp, plateau)

    # If a mode dropped out during NNLS, remove its dtau entry too
    if nlls_success and estimate_error and len(tau) < len(tau0):
        n_kill = 0
        for i in range(len(tau0)):
            if np.min(np.abs(tau0[i] - tau)) > 1e-12 * tau0[i]:
                dtau   = np.delete(dtau, i - n_kill)
                n_kill += 1

    if not nlls_success:
        dtau = np.full(len(tau), np.nan)

    return g, tau, dtau
