"""
plotting.py — Plotting routines for pyReSpect.

All plot functions are pure in the sense that they only read from
result dataclasses and return matplotlib Figure objects. No computation
happens here. Figures can optionally be saved to disk.

Public API
----------
plot(which, toFile, path, t, Gt, weights, cont_result, disc_result)
    Dispatcher: produces all requested figures and returns them as a list.

All other functions are private to this module.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .config import ReSpectError
from .continuous import ContinuousResult
from .discrete import DiscreteResult


# Valid 'which' tokens
_VALID_WHICH = {"base", "full"}

# Both tokens require both results
_NEEDS_CONT = {"base", "full"}
_NEEDS_DISC = {"base", "full"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot(
    which:       Union[str, list[str]],
    toFile:      bool,
    path:        str,
    t:           np.ndarray,
    Gt:          np.ndarray,
    weights:     np.ndarray,
    cont_result: Optional[ContinuousResult],
    disc_result: Optional[DiscreteResult],
) -> list[Figure]:
    """Produce plots of the fitted spectra.

    Parameters
    ----------
    which : str or list of str
        Which plots to produce. Valid values:

        - ``"base"`` : single figure with two panels —
                       discrete modes g_i/τ_i overlaid on H(s),
                       G(t) data vs continuous and discrete fits.
        - ``"full"`` : above + diagnostic figure with three panels —
                       log p(λ) vs λ, ρ-η L-curve, AIC scan.
                       Diagnostic panels requiring L-curve data are
                       silently omitted when lam_C was pre-specified.

    toFile : bool
        If True, save each figure to path.
    path : str
        Output directory for saved figures.
    t : np.ndarray, shape (n,)
        Experimental time points.
    Gt : np.ndarray, shape (n,)
        Experimental relaxation modulus G(t).
    weights : np.ndarray, shape (n,)
        Per-datapoint weights.
    cont_result : ContinuousResult or None
    disc_result : DiscreteResult or None

    Returns
    -------
    figs : list of Figure
        All figures produced, in the order requested.

    Raises
    ------
    ReSpectError
        If a requested plot requires a result that is not available,
        or if an invalid 'which' token is supplied.
    """
    tokens = _parse_which(which)
    _validate_which(tokens, cont_result, disc_result)

    if toFile:
        os.makedirs(path, exist_ok=True)

    figs: list[Figure] = []

    for token in tokens:

        if token == "base":
            fig = _plot_base(t, Gt, cont_result, disc_result)
            figs.append(fig)
            if toFile:
                _save_fig(fig, path, "Gfit.pdf")
            else:
                plt.show()

        elif token == "full":
            fig1 = _plot_base(t, Gt, cont_result, disc_result)
            figs.append(fig1)
            if toFile:
                _save_fig(fig1, path, "Gfit.pdf")
            else:
                plt.show()

            fig2 = _plot_diagnostics(cont_result, disc_result)
            figs.append(fig2)
            if toFile:
                _save_fig(fig2, path, "diagnostics.pdf")
            else:
                plt.show()
                
    return figs


# ---------------------------------------------------------------------------
# Private: main figure
# ---------------------------------------------------------------------------

def _plot_base(
    t:           np.ndarray,
    Gt:          np.ndarray,
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> Figure:
    """Two-panel figure: g_i/τ_i, exp(H) overlay | G(t) fits.

    Left panel
        Discrete mode weights g_i plotted against relaxation times τ_i,
        overlaid on the continuous spectrum exp(H(s)).

    Right panel
        Experimental G(t) data against both the continuous and discrete
        model fits on a log-log axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Left: discrete modes overlaid on continuous spectrum ----
    ax = axes[0]
    ax.loglog(
        cont_result.s, np.exp(cont_result.H),
        label='CRS',
    )
    ax.loglog(
        disc_result.tau, disc_result.g,
        'o-', label='DRS',
    )
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$g, h(\tau)$')
    ax.legend()

    # ---- Right: G(t) data vs continuous and discrete fits ----
    ax = axes[1]
    ax.loglog(t, Gt, 'x', label='data', c='gray')
    ax.loglog(t, cont_result.G_fit, label='continuous fit')
    ax.loglog(t, disc_result.G_fit, '--', label='discrete fit')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$G(t)$')
    ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private: diagnostic figure
# ---------------------------------------------------------------------------

def _plot_diagnostics(
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> Figure:
    """Three-panel diagnostic figure: log p(λ) | ρ-η L-curve | AIC scan.

    The left and middle panels require L-curve data and are replaced by
    empty labelled axes when lam_C was pre-specified (cont_result.lam
    is None). The AIC panel is always shown.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ---- Left: log p(λ) ----
    ax = axes[0]
    if cont_result.lam is not None:
        ax.plot(cont_result.lam, cont_result.log_P, 'o-')
        ax.axvline(x=cont_result.lam_C, color='gray', label=r'$\lambda_M$')
        ax.set_xscale('log')
        ax.set_ylim(-20, 1)
        ax.legend(loc='upper left')
    else:
        ax.text(0.5, 0.5, 'L-curve not computed\n(lam_C pre-specified)',
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\log\, p(\lambda)$')

    # ---- Middle: ρ-η L-curve ----
    ax = axes[1]
    if cont_result.lam is not None:
        ax.plot(cont_result.rho, cont_result.eta)
        ax.scatter(cont_result.rho, cont_result.eta, marker='x')

        rho_opt = np.exp(np.interp(
            np.log(cont_result.lam_C),
            np.log(cont_result.lam),
            np.log(cont_result.rho),
        ))
        eta_opt = np.exp(np.interp(
            np.log(cont_result.lam_C),
            np.log(cont_result.lam),
            np.log(cont_result.eta),
        ))
        ax.plot(rho_opt, eta_opt, 'o', color='C1', label=r'$\lambda_M$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'L-curve not computed\n(lam_C pre-specified)',
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\eta$')

    # ---- Right: AIC scan ----
    ax = axes[2]
    color_aic = 'C2'
    ax.plot(disc_result.wt_base, disc_result.AIC_bst,
            color=color_aic, label='AIC')
    ax.set_xlabel(r'$w_b$')
    ax.set_ylabel('AIC', color=color_aic)
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelcolor=color_aic)

    ax2 = ax.twinx()
    color_n = 'C1'
    ax2.plot(disc_result.wt_base, disc_result.N_bst,
             color=color_n, linestyle='--', label=r'$N_\mathrm{bst}$')
    ax2.set_ylabel(r'$N_\mathrm{bst}$', color=color_n)
    ax2.tick_params(axis='y', labelcolor=color_n)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private: helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: Figure, path: str, fname: str) -> None:
    """Save a figure to path/fname."""
    fig.savefig(os.path.join(path, fname))


def _parse_which(which: Union[str, list[str]]) -> list[str]:
    """Normalise 'which' to a list and validate tokens."""
    tokens = [which] if isinstance(which, str) else list(which)
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
    """Raise ReSpectError if a required result is missing."""
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
