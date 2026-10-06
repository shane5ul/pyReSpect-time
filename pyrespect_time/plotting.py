"""
plotting.py — Plotting routines for pyReSpect-time.

Plot functions only read from the result dataclasses and return
matplotlib Figure objects. No computation happens here.

Public API
----------
plot(which, toFile, path, t, Gt, cont_result, disc_result)
    Dispatcher: produces all requested figures and returns them as a list.

The figures have the same layout, labels and file names as those of
pyrespect_freq; only the data panel differs.
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
from .io import _parse_which, _validate_which


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot(
    which:       Union[str, list[str]],
    toFile:      bool,
    path:        Union[str, os.PathLike],
    t:           np.ndarray,
    Gt:          np.ndarray,
    cont_result: Optional[ContinuousResult],
    disc_result: Optional[DiscreteResult],
) -> list[Figure]:
    """Produce plots of the fitted spectra.

    Parameters
    ----------
    which : str or list of str
        Which plots to produce. Valid values:

        - ``"base"`` : two-panel figure —
                       left:  exp(H(s)) with ±2.5 dH error band and
                              discrete modes g_i overlaid;
                       right: G(t) data vs continuous and discrete fits.
        - ``"full"`` : above + three-panel diagnostic figure —
                       log p(λ) vs λ, ρ-η L-curve, AIC scan.
                       Panels requiring L-curve data show a note instead
                       when lam_C was pre-specified.

    toFile : bool
        If True, save each figure as a PDF in *path* (Gfit.pdf,
        diagnostics.pdf) and close it. If False, show it on screen.
    path : str or path-like
        Output directory for saved figures.
    t : np.ndarray, shape (n,)
        Experimental time points.
    Gt : np.ndarray, shape (n,)
        Experimental relaxation modulus G(t).
    cont_result : ContinuousResult or None
    disc_result : DiscreteResult or None

    Returns
    -------
    figs : list of Figure
        All figures produced, base figure first.

    Raises
    ------
    ReSpectError
        If a requested plot requires a result that is not available,
        or if an invalid 'which' token is supplied.
    """
    tokens = _parse_which(which)
    _validate_which(tokens, cont_result, disc_result)

    figs  = [_plot_base(t, Gt, cont_result, disc_result)]
    names = ["Gfit.pdf"]

    if "full" in tokens:
        figs.append(_plot_diagnostics(cont_result, disc_result))
        names.append("diagnostics.pdf")

    if toFile:
        os.makedirs(path, exist_ok=True)
        for fig, name in zip(figs, names):
            fig.savefig(os.path.join(path, name))
            plt.close(fig)
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
    """Two-panel figure: spectra | fits to the data."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    _plot_spectra(axes[0], cont_result, disc_result)

    # ---- Right: G(t) data vs continuous and discrete fits ----
    ax = axes[1]
    ax.loglog(t, Gt, "x", c="gray", label="data")
    ax.loglog(t, cont_result.G_fit, "-",  c="C0", label="CRS")
    ax.loglog(t, disc_result.G_fit, "--", c="C1", label="DRS")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$G(t)$")
    ax.legend()

    fig.tight_layout()
    return fig


def _plot_spectra(
    ax,
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> None:
    """Left panel: exp(H(s)) with error band, overlaid with g_i vs τ_i."""
    ax.loglog(cont_result.s, np.exp(cont_result.H), c="C0", label="CRS")

    if cont_result.dH is not None:
        for sign in (+1, -1):
            ax.loglog(
                cont_result.s,
                np.exp(cont_result.H + sign * 2.5 * cont_result.dH),
                c="gray", alpha=0.5,
            )

    ax.loglog(disc_result.tau, disc_result.g, "o-", c="C1", label="DRS")
    ax.set_xlabel(r"$s,\ \tau_i$")
    ax.set_ylabel(r"$h(s),\ g_i$")
    ax.legend()


# ---------------------------------------------------------------------------
# Private: diagnostic figure
# ---------------------------------------------------------------------------

def _plot_diagnostics(
    cont_result: ContinuousResult,
    disc_result: DiscreteResult,
) -> Figure:
    """Three-panel diagnostic figure: log p(λ) | ρ-η L-curve | AIC scan.

    The left and middle panels show a note instead when lam_C was
    pre-specified (cont_result.lam is None). The AIC panel is always
    shown.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    lam_label = r"$\lambda_C = {:.2e}$".format(cont_result.lam_C)
    no_lcurve = "L-curve not computed\n(lam_C pre-specified)"

    # ---- Left: log p(λ) ----
    ax = axes[0]
    if cont_result.lam is not None:
        ax.plot(cont_result.lam, cont_result.log_P, "o-")
        ax.axvline(cont_result.lam_C, color="gray", label=lam_label)
        ax.set_xscale("log")
        ax.set_ylim(-20, 1)
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, no_lcurve,
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\log\, p(\lambda)$")

    # ---- Middle: ρ-η L-curve ----
    ax = axes[1]
    if cont_result.lam is not None:
        ax.plot(cont_result.rho, cont_result.eta, "x-")

        # interpolate the chosen point in log-log space
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
        ax.plot(rho_opt, eta_opt, "o", c="C1", label=lam_label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, no_lcurve,
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\eta$")

    # ---- Right: AIC scan ----
    ax  = axes[2]
    ax2 = ax.twinx()

    color_aic = "C2"
    color_n   = "C1"

    ax.plot(disc_result.wt_base, disc_result.AIC_bst,
            color=color_aic, label="AIC")
    ax.set_xlabel(r"$w_b$")
    ax.set_ylabel("AIC", color=color_aic)
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelcolor=color_aic)

    ax2.plot(disc_result.wt_base, disc_result.N_bst,
             color=color_n, linestyle="--", label=r"$N_\mathrm{bst}$")
    ax2.set_ylabel(r"$N_\mathrm{bst}$", color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    fig.tight_layout()
    return fig
