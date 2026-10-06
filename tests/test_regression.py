"""
Regression tests: the spectra computed for every data file in tests/ must
match the stored baselines in tests/baseline/.

The baselines pin the numerical output of the package. If an intentional
change to the algorithm moves the numbers, regenerate them with

    python tests/make_baseline.py

and review the differences before committing.
"""
from pathlib import Path

import numpy as np
import pytest

from pyrespect_time import ReSpect, ReSpectConfig

TESTS    = Path(__file__).parent
BASELINE = TESTS / "baseline"

# (data file, config overrides)
CASES = [(f"test{i}", {}) for i in range(1, 8)] + [("test7", {"plateau": True})]

# Tolerances. Results are reproducible to round-off on one machine; these
# leave room for differences between platforms and BLAS libraries.
RTOL_FIT   = 1e-5     # reconstructed moduli
RTOL_MODES = 1e-3     # lam_C, tau_i, g_i, G0
ATOL_H     = 1e-3     # H(s) is a logarithm, so this is a relative error in h


def tag(name, kwargs):
    return name + ("_plateau" if kwargs.get("plateau") else "")


def run_case(name, kwargs):
    """Fit one data file and return the quantities that are pinned."""
    solver = ReSpect(ReSpectConfig(**kwargs)).fit(TESTS / f"{name}.dat")
    c, d = solver.continuous, solver.discrete
    return dict(
        s=c.s, H=c.H, lam_C=c.lam_C, G0_cont=c.G0, G_fit_cont=c.G_fit,
        g=d.g, tau=d.tau, N=d.N, G0_disc=d.G0, G_fit_disc=d.G_fit,
    )


@pytest.mark.parametrize("name, kwargs", CASES, ids=[tag(*c) for c in CASES])
def test_matches_baseline(name, kwargs):
    got = run_case(name, kwargs)
    ref = np.load(BASELINE / f"{tag(name, kwargs)}.npz")

    # continuous spectrum
    np.testing.assert_allclose(got["s"], ref["s"], rtol=1e-12)
    np.testing.assert_allclose(got["lam_C"], ref["lam_C"], rtol=RTOL_MODES)
    np.testing.assert_allclose(got["H"], ref["H"], atol=ATOL_H)
    np.testing.assert_allclose(got["G_fit_cont"], ref["G_fit_cont"], rtol=RTOL_FIT)
    np.testing.assert_allclose(got["G0_cont"], ref["G0_cont"], rtol=RTOL_MODES)

    # discrete spectrum
    assert got["N"] == int(ref["N"])
#    np.testing.assert_allclose(got["tau"], ref["tau"], rtol=RTOL_MODES)
#    np.testing.assert_allclose(got["g"], ref["g"], rtol=RTOL_MODES)
    np.testing.assert_allclose(got["G_fit_disc"], ref["G_fit_disc"], rtol=RTOL_FIT)
    np.testing.assert_allclose(got["G0_disc"], ref["G0_disc"], rtol=RTOL_MODES, atol=1e-12)
