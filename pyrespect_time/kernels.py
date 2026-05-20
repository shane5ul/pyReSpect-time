"""
kernels.py — Pure, stateless kernel functions for pyReSpect.

These functions implement the discretized form of the kernel operator

    K[H](t) = ∫ exp(H(s)) exp(-t/s) d ln s  ≈  kernMat @ exp(H)

and its Jacobian. They are shared by both the continuous and discrete
spectrum solvers and have no side effects.

Functions
---------
get_kern_mat(s, t)
    Pre-store the weighted kernel matrix W * exp(-T/S).

kernel_prestore(H, kern_mat, G0=None)
    Evaluate K[H](t) = kernMat @ exp(H) + G0.

kernel_D(H, kern_mat)
    Evaluate the Jacobian dK_i/dH_j = kernMat * exp(H_j).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def get_kern_mat(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pre-store the weighted kernel matrix for fast kernel evaluation.

    Computes the n × ns matrix

        K_{ij} = h_j * exp(-t_i / s_j)

    where h_j are the trapezoidal quadrature weights in log-space:

        h_j = Δ(ln s)_j / 2

    Multiplying this matrix by exp(H) approximates the continuous integral

        G(t_i) = ∫ exp(H(s)) exp(-t_i/s) d ln s

    Parameters
    ----------
    s : np.ndarray, shape (ns,)
        Relaxation mode axis (must be positive and monotonically increasing).
    t : np.ndarray, shape (n,)
        Experimental time points.

    Returns
    -------
    kern_mat : np.ndarray, shape (n, ns)
        Weighted kernel matrix  h * exp(-T/S).
    """
    ns = len(s)

    # Trapezoidal weights in log-space: h_j = Δ(ln s)_j / 2
    hs = np.empty(ns)
    hs[0]      = 0.5 * np.log(s[1] / s[0])
    hs[-1]     = 0.5 * np.log(s[-1] / s[-2])
    hs[1:-1]   = 0.5 * (np.log(s[2:]) - np.log(s[:-2]))

    S, T = np.meshgrid(s, t)                 # both shape (n, ns)
    return np.exp(-T / S) * hs               # (n, ns)


def kernel_prestore(
    H: np.ndarray,
    kern_mat: np.ndarray,
    G0: Optional[float] = None,
) -> np.ndarray:
    """Evaluate the kernel operator K[H](t) using a pre-stored kernel matrix.

    Computes

        G(t) = kernMat @ exp(H) + G0

    which approximates

        G(t_i) = ∫ exp(H(s)) exp(-t_i/s) d ln s  +  G0

    Parameters
    ----------
    H : np.ndarray, shape (ns,)
        Log relaxation spectrum H(s).
    kern_mat : np.ndarray, shape (n, ns)
        Pre-stored weighted kernel matrix from get_kern_mat().
    G0 : float or None, optional
        Plateau modulus. Added as a constant offset when provided.
        Default: None (no plateau, equivalent to G0 = 0).

    Returns
    -------
    G : np.ndarray, shape (n,)
        Model relaxation modulus G(t).
    """
    G = kern_mat @ np.exp(H)
    if G0 is not None:
        G = G + G0
    return G


def kernel_D(H: np.ndarray, kern_mat: np.ndarray) -> np.ndarray:
    """Evaluate the Jacobian of the kernel operator with respect to H.

    Computes the n × ns matrix

        J_{ij} = dK_i / dH_j = kernMat_{ij} * exp(H_j)

    This approximation follows from the chain rule applied to
    K[H](t_i) = Σ_j kernMat_{ij} * exp(H_j).

    Parameters
    ----------
    H : np.ndarray, shape (ns,)
        Log relaxation spectrum H(s).
    kern_mat : np.ndarray, shape (n, ns)
        Pre-stored weighted kernel matrix from get_kern_mat().

    Returns
    -------
    J : np.ndarray, shape (n, ns)
        Jacobian matrix dK/dH.
    """
    n, ns = kern_mat.shape

    # Broadcast exp(H) across all n rows: each column j scaled by exp(H_j)
    exp_H = np.exp(H).reshape(1, ns)          # (1, ns)
    return kern_mat * exp_H                    # (n, ns)
