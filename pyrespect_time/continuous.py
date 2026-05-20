"""
continuous.py — Continuous relaxation spectrum solver for pyReSpect.

Extracts the continuous relaxation spectrum H(s) from stress relaxation
data G(t) by solving the regularized inverse problem:

    min_{H} || w * (G(t) - K[H](t)) / G(t) ||^2  +  λ || L H ||^2

where K is the kernel operator, L is the second-difference matrix, and
λ is the regularization parameter determined via a Bayesian L-curve method.

Public API
----------
fit_continuous(t, Gt, weights, config) -> ContinuousResult

All other functions are private to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .config import ReSpectConfig
from .kernels import get_kern_mat, kernel_prestore, kernel_D


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContinuousResult:
    """Results from the continuous spectrum fit.

    Attributes
    ----------
    s : np.ndarray, shape (ns,)
        Relaxation mode axis.
    H : np.ndarray, shape (ns,)
        Log relaxation spectrum H(s).
    lam_C : float
        Regularization parameter λ used for the final fit.
    G_fit : np.ndarray, shape (n,)
        Reconstructed G(t) from the continuous spectrum.
    G0 : float or None
        Plateau modulus. None if config.plateau is False.
    lam : np.ndarray or None
        Array of λ values scanned on the L-curve.
        None if lam_C was pre-specified in config.
    rho : np.ndarray or None
        Residual norm at each λ. None if lam_C was pre-specified.
    eta : np.ndarray or None
        Roughness norm at each λ. None if lam_C was pre-specified.
    log_P : np.ndarray or None
        Normalized log-probability log p(λ) at each λ.
        None if lam_C was pre-specified.
    """
    s:      np.ndarray
    H:      np.ndarray
    lam_C:  float
    G_fit:  np.ndarray
    G0:     Optional[float]

    # L-curve diagnostics — None when lam_C is pre-specified
    lam:    Optional[np.ndarray]
    rho:    Optional[np.ndarray]
    eta:    Optional[np.ndarray]
    log_P:  Optional[np.ndarray]
    H_lam:  Optional[np.ndarray]  # shape (ns, n_lam); H snapshots at each λ


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_continuous(
    t:       np.ndarray,
    Gt:      np.ndarray,
    weights: np.ndarray,
    config:  ReSpectConfig,
) -> ContinuousResult:
    """Fit the continuous relaxation spectrum H(s) to stress relaxation data.

    Solves the Tikhonov-regularized inverse problem and determines the
    optimal regularization parameter λ via a Bayesian L-curve method
    (or uses a pre-specified value from config).

    Parameters
    ----------
    t : np.ndarray, shape (n,)
        Experimental time points.
    Gt : np.ndarray, shape (n,)
        Experimental relaxation modulus G(t).
    weights : np.ndarray, shape (n,)
        Per-datapoint weights. Pass np.ones(n) for uniform weighting.
    config : ReSpectConfig
        Configuration object.

    Returns
    -------
    ContinuousResult
    """
    # ------------------------------------------------------------------
    # Build the relaxation mode axis s
    # ------------------------------------------------------------------
    n    = len(t)
    ns   = config.ns
    fei  = config.freq_end_int          # 1, 2, or 3

    tmin, tmax = t[0], t[-1]

    if fei == 1:
        smin = np.exp(-np.pi / 2) * tmin
        smax = np.exp( np.pi / 2) * tmax
    elif fei == 2:
        smin, smax = tmin, tmax
    else:  # fei == 3
        smin = np.exp( np.pi / 2) * tmin
        smax = np.exp(-np.pi / 2) * tmax

    hs_ratio = (smax / smin) ** (1.0 / (ns - 1))
    s        = smin * hs_ratio ** np.arange(ns)

    kern_mat = get_kern_mat(s, t)

    # ------------------------------------------------------------------
    # Initial guess for H (and G0 if plateau)
    # ------------------------------------------------------------------
    H, G0 = _initialize_H(Gt, weights, s, kern_mat, config)

    # ------------------------------------------------------------------
    # Determine λ
    # ------------------------------------------------------------------
    if config.lam_C is None:
        lam_C, lam, rho, eta, log_P, H_lam = _lcurve(
            Gt, weights, H, kern_mat, config, G0
        )
        # Update H (and G0) at the optimal λ
        H, G0 = _get_H(lam_C, Gt, weights, H, kern_mat, config.plateau, G0)
    else:
        lam_C                        = config.lam_C
        lam = rho = eta = log_P = H_lam = None
        H, G0 = _get_H(lam_C, Gt, weights, H, kern_mat, config.plateau, G0)

    # ------------------------------------------------------------------
    # Reconstruct G(t)
    # ------------------------------------------------------------------
    G_fit = kernel_prestore(H, kern_mat, G0)

    return ContinuousResult(
        s=s,
        H=H,
        lam_C=lam_C,
        G_fit=G_fit,
        G0=G0,
        lam=lam,
        rho=rho,
        eta=eta,
        log_P=log_P,
        H_lam=H_lam,
    )


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _initialize_H(
    Gexp:     np.ndarray,
    wexp:     np.ndarray,
    s:        np.ndarray,
    kern_mat: np.ndarray,
    config:   ReSpectConfig,
) -> tuple[np.ndarray, Optional[float]]:
    """Compute an initial guess for H (and optionally G0).

    Uses a large regularization parameter so the solution is dominated
    by the smoothness prior — a safe, featureless starting point.

    Parameters
    ----------
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    s : np.ndarray, shape (ns,)
    kern_mat : np.ndarray, shape (n, ns)
    config : ReSpectConfig

    Returns
    -------
    H : np.ndarray, shape (ns,)
    G0 : float or None
    """
    ns   = len(s)
    H    = -5.0 * np.ones(ns) + np.sin(np.pi * s)
    lam  = 1e0

    if config.plateau:
        G0_guess      = float(np.min(Gexp))
        H, G0 = _get_H(lam, Gexp, wexp, H, kern_mat, plateau=True, G0=G0_guess)
        return H, G0
    else:
        H = _get_H(lam, Gexp, wexp, H, kern_mat, plateau=False, G0=None)[0]
        return H, None


def _get_H(
    lam:      float,
    Gexp:     np.ndarray,
    wexp:     np.ndarray,
    H:        np.ndarray,
    kern_mat: np.ndarray,
    plateau:  bool,
    G0:       Optional[float],
) -> tuple[np.ndarray, Optional[float]]:
    """Find H_λ that minimizes the Tikhonov functional V(λ).

    V(λ) = || w * (G_exp - K[H]) / G_exp ||^2  +  λ || L H ||^2

    Uses the Levenberg-Marquardt algorithm with an analytic Jacobian
    for efficiency.

    Parameters
    ----------
    lam : float
        Regularization parameter λ.
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    H : np.ndarray, shape (ns,)
        Initial guess.
    kern_mat : np.ndarray, shape (n, ns)
    plateau : bool
        If True, G0 is appended to the optimization vector.
    G0 : float or None
        Current estimate of plateau modulus (used when plateau=True).

    Returns
    -------
    H_opt : np.ndarray, shape (ns,)
    G0_opt : float or None
    """
    if plateau and G0 is not None:
        # Concatenate H and G0 into a single vector for the optimizer
        x0  = np.append(H, G0)
        res = least_squares(
            _residual_LM, x0,
            jac=_jacobian_LM,
            args=(lam, Gexp, wexp, kern_mat),
        )
        return res.x[:-1], float(res.x[-1])
    else:
        res = least_squares(
            _residual_LM, H,
            jac=_jacobian_LM,
            args=(lam, Gexp, wexp, kern_mat),
        )
        return res.x, None


def _residual_LM(
    x:        np.ndarray,
    lam:      float,
    Gexp:     np.ndarray,
    wexp:     np.ndarray,
    kern_mat: np.ndarray,
) -> np.ndarray:
    """Residual vector for the TRF method optimizer.

    Returns the concatenated vector

        r = [ w * (1 - K[H](t) / G_exp),   sqrt(λ) * L H ]

    of length n + (ns - 2). If G0 is appended to x, it is unpacked
    and added as a constant offset to K[H].

    Parameters
    ----------
    x : np.ndarray
        Optimization vector. Either H (shape ns,) or [H, G0] (shape ns+1,).
    lam : float
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    kern_mat : np.ndarray, shape (n, ns)

    Returns
    -------
    r : np.ndarray, shape (n + ns - 2,)
    """
    n, ns = kern_mat.shape
    nl    = ns - 2

    # Unpack G0 if plateau
    if len(x) > ns:
        H  = x[:-1]
        G0 = float(x[-1])
    else:
        H  = x
        G0 = None

    r       = np.empty(n + nl)
    G_model = kernel_prestore(H, kern_mat, G0)
    r[:n]   = wexp * (1.0 - G_model / Gexp)
    r[n:]   = np.sqrt(lam) * np.diff(H, n=2)   # curvature penalty

    return r


def _jacobian_LM(
    x:        np.ndarray,
    lam:      float,
    Gexp:     np.ndarray,
    wexp:     np.ndarray,
    kern_mat: np.ndarray,
) -> np.ndarray:
    """Analytic Jacobian for the Levenberg-Marquardt optimizer.

    Returns Jr of shape (n + ns - 2, ns) or (n + ns - 2, ns + 1)
    when G0 is included.

    Parameters
    ----------
    x : np.ndarray
        Optimization vector. Either H (ns,) or [H, G0] (ns+1,).
    lam : float
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    kern_mat : np.ndarray, shape (n, ns)

    Returns
    -------
    Jr : np.ndarray, shape (n + ns - 2, ns) or (n + ns - 2, ns + 1)
    """
    n, ns = kern_mat.shape
    nl    = ns - 2

    # Second-difference matrix L: (nl × ns) tridiagonal [1, -2, 1]
    L = (
        np.diag(np.ones(ns - 1),  1)
        + np.diag(np.ones(ns - 1), -1)
        + np.diag(-2.0 * np.ones(ns))
    )[1:nl + 1, :]

    # Weight matrix broadcast: (n, ns)
    W_mat = (wexp / Gexp).reshape(n, 1) * np.ones((1, ns))

    if len(x) > ns:
        H  = x[:-1]
        Jr = np.zeros((n + nl, ns + 1))

        Jr[:n,  :ns] = -kernel_D(H, kern_mat) * W_mat
        Jr[:n,   ns] = -wexp / Gexp            # ∂r_i/∂G0
        Jr[n:,  :ns] = np.sqrt(lam) * L
        Jr[n:,   ns] = 0.0                     # curvature doesn't depend on G0
    else:
        H  = x
        Jr = np.zeros((n + nl, ns))

        Jr[:n, :ns] = -kernel_D(H, kern_mat) * W_mat
        Jr[n:, :ns] = np.sqrt(lam) * L

    return Jr


def _get_A_matrix(ns: int) -> np.ndarray:
    """Build the symmetric matrix A = L^T L for Bayesian error analysis.

    L is the (ns-2) × ns second-difference matrix.

    Parameters
    ----------
    ns : int
        Number of relaxation modes.

    Returns
    -------
    A : np.ndarray, shape (ns, ns)
    """
    nl = ns - 2
    L  = (
        np.diag(np.ones(ns - 1),  1)
        + np.diag(np.ones(ns - 1), -1)
        + np.diag(-2.0 * np.ones(ns))
    )[1:nl + 1, :]
    return L.T @ L


def _get_B_matrix(
    H:        np.ndarray,
    kern_mat: np.ndarray,
    Gexp:     np.ndarray,
    wexp:     np.ndarray,
    G0:       Optional[float],
) -> np.ndarray:
    """Build the matrix B = J^T J + diag(r^T J) for Bayesian error analysis.

    Parameters
    ----------
    H : np.ndarray, shape (ns,)
    kern_mat : np.ndarray, shape (n, ns)
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    G0 : float or None

    Returns
    -------
    B : np.ndarray, shape (ns, ns)
    """
    n, ns = kern_mat.shape

    W_mat = (wexp / Gexp).reshape(n, 1) * np.ones((1, ns))
    Jr    = -kernel_D(H, kern_mat) * W_mat          # (n, ns)

    r     = wexp * (1.0 - kernel_prestore(H, kern_mat, G0) / Gexp)

    return Jr.T @ Jr + np.diag(r @ Jr)


def _lcurve(
    Gexp:     np.ndarray,
    wexp:     np.ndarray,
    H:        np.ndarray,
    kern_mat: np.ndarray,
    config:   ReSpectConfig,
    G0:       Optional[float],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan λ and determine the optimal regularization parameter λ_M.

    Uses a Bayesian formulation: computes log p(λ) at each grid point
    and estimates λ_M as the log-weighted mean. Scanning proceeds from
    large λ to small and terminates early when log p drops 18 below
    its running maximum.

    Parameters
    ----------
    Gexp : np.ndarray, shape (n,)
    wexp : np.ndarray, shape (n,)
    H : np.ndarray, shape (ns,)
        Used as warm-start; updated each iteration
    kern_mat : np.ndarray, shape (n, ns)
    config : ReSpectConfig
    G0 : float or None

    Returns
    -------
    lam_M : float
        Optimal regularization parameter.
    lam : np.ndarray
        λ values scanned (truncated to significant range).
    rho : np.ndarray
        Residual norms at each λ.
    eta : np.ndarray
        Roughness norms at each λ.
    log_P : np.ndarray
        Normalized log p(λ) at each λ.
    H_lam : np.ndarray, shape (ns, n_lam)
        H snapshots at each λ (truncated to significant range).
    """
    ns = len(H)

    # Build the λ grid: lam_density points per decade, large → small
    npoints = int(
        config.lam_density
        * (np.log10(config.lam_max) - np.log10(config.lam_min))
    )
    hlam    = (config.lam_max / config.lam_min) ** (1.0 / (npoints - 1))
    lam_grid = config.lam_min * hlam ** np.arange(npoints)

    rho     = np.zeros(npoints)
    eta     = np.zeros(npoints)
    log_P   = np.zeros(npoints)
    H_lam   = np.zeros((ns, npoints))

    # Precompute A matrix for Bayesian evidence
    A_mat              = _get_A_matrix(ns)
    _, log_det_N       = np.linalg.slogdet(A_mat)

    log_P_max = -np.inf
    i_start   = 0                           # will track early-exit index

    for i in reversed(range(npoints)):
        lamb = lam_grid[i]

        H, G0 = _get_H(lamb, Gexp, wexp, H, kern_mat, config.plateau, G0)

        G_model  = kernel_prestore(H, kern_mat, G0)
        rho[i]   = np.linalg.norm(wexp * (1.0 - G_model / Gexp))
        eta[i]   = np.linalg.norm(np.diff(H, n=2))
        H_lam[:, i] = H

        B_mat              = _get_B_matrix(H, kern_mat, Gexp, wexp, G0)
        _, log_det_C       = np.linalg.slogdet(lamb * A_mat + B_mat)

        V        = rho[i] ** 2 + lamb * eta[i] ** 2
        log_P[i] = (
            -V
            + 0.5 * (log_det_N + ns * np.log(lamb) - log_det_C)
            - lamb
        )

        if log_P[i] > log_P_max:
            log_P_max = log_P[i]
        elif log_P[i] < log_P_max - 18:
            i_start = i
            break

    # Truncate to the significant λ range
    lam_grid = lam_grid[i_start:]
    log_P    = log_P[i_start:]
    rho      = rho[i_start:]
    eta      = eta[i_start:]
    H_lam    = H_lam[:, i_start:]

    # Normalize log_P
    log_P = log_P - np.max(log_P)

    # Bayesian estimate: λ_M = exp( E[log λ] ) under p(λ) ∝ exp(log_P)
    p_lam = np.exp(log_P)
    p_lam = p_lam / np.sum(p_lam)
    lam_M = np.exp(np.sum(p_lam * np.log(lam_grid)))

    # Apply smoothness nudge from config
    if config.SmFacLam > 0:
        lam_M = np.exp(
            np.log(lam_M)
            + config.SmFacLam * (np.max(np.log(lam_grid)) - np.log(lam_M))
        )
    elif config.SmFacLam < 0:
        lam_M = np.exp(
            np.log(lam_M)
            + config.SmFacLam * (np.log(lam_M) - np.min(np.log(lam_grid)))
        )

    return lam_M, lam_grid, rho, eta, log_P, H_lam
