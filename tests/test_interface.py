"""
Interface tests for pyrespect_time.

Apart from the block of package-specific definitions below, this file is
the same as tests/test_interface.py in pyReSpect-freq: the two packages
are meant to present the same interface.
"""
import dataclasses
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

# ---- package-specific definitions ------------------------------------------
import pyrespect_time as pkg
from pyrespect_time.io import load_data
from pyrespect_time.continuous import _error_band

DATA         = "test3.dat"            # raw data file used throughout
NCOLS        = 2                      # columns in a raw data file
N_WEIGHT_COL = 1                      # extra columns holding weights
X            = "t"                    # name of the abscissa attribute
DATA_ATTRS   = ("t", "Gt", "weights")
GFIT_HEADER  = "# t  G_cont  G_disc"
FIT_ARRAYS   = ["source", "Gt"]       # leading parameters of fit()
# -----------------------------------------------------------------------------

ReSpect, ReSpectConfig = pkg.ReSpect, pkg.ReSpectConfig
ReSpectError, ReSpectWarning = pkg.ReSpectError, pkg.ReSpectWarning

TESTS = Path(__file__).parent
ROOT  = TESTS.parent

CONT_FIELDS = ["s", "H", "G_fit", "G0", "lam_C",
               "lam", "rho", "eta", "log_P", "dH", "H_lam"]
DISC_FIELDS = ["g", "tau", "dtau", "N", "G_fit", "G0", "error",
               "wt_base", "AIC_bst", "N_bst", "nz_N_bst"]
BASE_FILES  = {"crs.dat", "drs.dat", "Gfit.dat"}
FULL_FILES  = BASE_FILES | {"rho-eta.dat", "logPlam.dat", "aic.dat"}


@pytest.fixture(scope="module")
def raw():
    """The raw data file as a 2-D array."""
    return np.loadtxt(TESTS / DATA)


@pytest.fixture(scope="module")
def solver():
    """A solver fitted to the raw data file with default settings."""
    return ReSpect().fit(TESTS / DATA)


def columns(data):
    return [data[:, j] for j in range(data.shape[1])]


def headers(fname):
    return [ln for ln in Path(fname).read_text().splitlines() if ln.startswith("#")]


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_exports():
    assert pkg.__all__ == ["ReSpect", "ReSpectConfig", "ReSpectError",
                           "ReSpectWarning", "ContinuousResult", "DiscreteResult"]
    assert isinstance(pkg.__version__, str)


def test_signatures():
    fit = inspect.signature(ReSpect.fit).parameters
    assert list(fit)[1:] == FIT_ARRAYS + ["weights", "resample"]
    assert fit["weights"].default is None and fit["resample"].default is None

    save = inspect.signature(ReSpect.save).parameters
    assert list(save)[1:] == ["which", "path"]
    assert (save["which"].default, save["path"].default) == ("base", "./")

    plot = inspect.signature(ReSpect.plot).parameters
    assert list(plot)[1:] == ["which", "toFile", "path"]
    assert (plot["which"].default, plot["toFile"].default,
            plot["path"].default) == ("base", False, "./")


def test_repr():
    assert repr(ReSpect()) == \
        "ReSpect(fitted=False, ns=100, plateau=False, freq_end='lenient')"


def test_unfitted_solver():
    s = ReSpect()
    assert s.continuous is None and s.discrete is None
    assert all(getattr(s, a) is None for a in DATA_ATTRS)
    for method in (s.save, s.plot):
        with pytest.raises(ReSpectError, match="before fit"):
            method()


# ---------------------------------------------------------------------------
# Input forms
# ---------------------------------------------------------------------------

def test_input_forms_are_equivalent(raw):
    cols = columns(raw)
    ref  = load_data(str(TESTS / DATA))
    for source in (TESTS / DATA, tuple(cols), list(cols), raw):
        for got, want in zip(load_data(source), ref):
            np.testing.assert_array_equal(got, want)


def test_fit_with_separate_arrays(raw, solver):
    s = ReSpect().fit(*columns(raw))
    np.testing.assert_array_equal(s.discrete.tau, solver.discrete.tau)
    np.testing.assert_array_equal(s.continuous.H, solver.continuous.H)


def test_data_are_sorted_and_deduplicated(raw):
    shuffled = np.vstack((raw[::-1], raw[:5]))
    for got, want in zip(load_data(shuffled, resample=False),
                         load_data(raw, resample=False)):
        np.testing.assert_array_equal(got, want)


def test_resample_default_and_switch(raw):
    n = len(raw)
    assert len(load_data(raw)[0]) == 100 != n
    assert len(load_data(raw, resample=False)[0]) == n
    assert len(load_data(raw, n_resample=40)[0]) == 40


def test_weights_three_ways(raw):
    cols = columns(raw)
    wt   = np.linspace(1.0, 2.0, len(raw))
    extra = [wt] * N_WEIGHT_COL
    ref  = load_data(tuple(cols), weights=wt)

    assert len(ref[0]) == len(raw)              # data with weights: not resampled
    assert np.allclose(ref[2].reshape(N_WEIGHT_COL, -1), wt)
    for source in (tuple(cols + extra), np.column_stack(cols + extra)):
        for got, want in zip(load_data(source), ref):
            np.testing.assert_array_equal(got, want)


def test_weights_errors(raw):
    cols = columns(raw)
    wt   = np.ones(len(raw))
    with pytest.raises(ReSpectError, match="supplied twice"):
        load_data(tuple(cols + [wt] * N_WEIGHT_COL), weights=wt)
    with pytest.raises(ReSpectError, match="shape"):
        load_data(tuple(cols), weights=wt[:-1])


@pytest.mark.parametrize("source", [
    "no-such-file.dat",
    (np.ones(5),),
    np.ones((5, NCOLS + N_WEIGHT_COL + 1)),
    42,
])
def test_bad_sources(source):
    with pytest.raises(ReSpectError):
        load_data(source)


def test_mismatched_arrays(raw):
    cols = columns(raw)
    cols[1] = cols[1][:-1]
    with pytest.raises(ReSpectError, match="same length"):
        load_data(tuple(cols))


def test_fit_resample_and_weights(raw):
    cols, n = columns(raw), len(raw)
    wt = np.linspace(1.0, 2.0, n)

    # config.resample is honoured, and fit(resample=...) overrides it
    s = ReSpect(ReSpectConfig(resample=False)).fit(TESTS / DATA)
    assert len(getattr(s, X)) == n
    assert len(getattr(ReSpect().fit(TESTS / DATA, resample=False), X)) == n

    # weights are kept, and data with weights are not resampled
    s = ReSpect().fit(*cols, weights=wt)
    assert len(getattr(s, X)) == n
    assert np.allclose(s.weights, wt)

    # asking for resampling explicitly is reported, not silently ignored
    with pytest.warns(ReSpectWarning, match="resample=True was ignored"):
        ReSpect().fit(*cols, weights=wt, resample=True)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def test_result_fields(solver):
    c, d = solver.continuous, solver.discrete
    assert [f.name for f in dataclasses.fields(c)] == CONT_FIELDS
    assert [f.name for f in dataclasses.fields(d)] == DISC_FIELDS

    ns = solver.config.ns
    assert c.s.shape == c.H.shape == c.dH.shape == (ns,)
    assert c.H_lam.shape == (ns, len(c.lam))
    assert len(c.lam) == len(c.rho) == len(c.eta) == len(c.log_P)
    assert isinstance(c.G0, float) and c.G0 == 0.0
    assert isinstance(c.lam_C, float)

    assert d.N == len(d.g) == len(d.tau) == len(d.dtau)
    assert np.all(np.diff(d.tau) > 0)
    assert isinstance(d.G0, float) and d.G0 == 0.0
    assert isinstance(d.error, float) and d.error >= 0.0
    assert len(d.wt_base) == len(d.AIC_bst) == len(d.N_bst) == len(d.nz_N_bst)
    assert c.G_fit.shape == d.G_fit.shape


def test_data_attributes(solver):
    n = len(getattr(solver, X))
    for name in DATA_ATTRS:
        assert getattr(solver, name).shape[-1] == n


def test_fixed_lambda_has_no_lcurve():
    c = ReSpect(ReSpectConfig(lam_C=1e-2)).fit(TESTS / DATA).continuous
    assert c.lam_C == 1e-2
    assert all(getattr(c, f) is None
               for f in ("lam", "rho", "eta", "log_P", "dH", "H_lam"))


def test_plateau_gives_float_G0():
    s = ReSpect(ReSpectConfig(plateau=True)).fit(TESTS / "test7.dat")
    assert isinstance(s.continuous.G0, float) and s.continuous.G0 > 0.0
    assert isinstance(s.discrete.G0, float)


def test_error_band_on_fine_lambda_grid():
    # no single lambda carries p > 0.1: used to give NaN
    H_lam = np.vstack((np.linspace(0.0, 1.0, 50), np.ones(50)))
    dH    = _error_band(H_lam, np.full(50, 1.0 / 50))
    assert np.all(np.isfinite(dH))
    assert dH[0] > 0.0 and dH[1] == 0.0


def test_close_modes_are_merged(solver):
    spacing = solver.discrete.tau[1:] / solver.discrete.tau[:-1]
    wide    = 1.5 * spacing.min()
    merged  = ReSpect(ReSpectConfig(min_tau_spacing=wide)).fit(TESTS / DATA)
    assert merged.discrete.N < solver.discrete.N


def test_single_n_scan_warns():
    with pytest.warns(ReSpectWarning, match="max_num_modes"):
        d = ReSpect(ReSpectConfig(max_num_modes=1)).fit(TESTS / DATA).discrete
    assert d.N == 1


# ---------------------------------------------------------------------------
# save and plot
# ---------------------------------------------------------------------------

def test_save_base_and_full(solver, tmp_path):
    assert solver.save(which="base", path=tmp_path / "base") is solver
    assert {p.name for p in (tmp_path / "base").iterdir()} == BASE_FILES

    solver.save("full", tmp_path / "full")
    out = tmp_path / "full"
    assert {p.name for p in out.iterdir()} == FULL_FILES

    expected = {
        "crs.dat":     ("# s  h", 2),
        "drs.dat":     ("# g  tau  dtau", 3),
        "Gfit.dat":    (GFIT_HEADER, len(GFIT_HEADER.split()) - 1),
        "rho-eta.dat": ("# lambda  rho  eta", 3),
        "logPlam.dat": ("# lambda  logP", 2),
        "aic.dat":     ("# wt_base  N_bst  AIC  nz_N_bst", 4),
    }
    for fname, (header, ncol) in expected.items():
        assert headers(out / fname) == [header]
        assert np.loadtxt(out / fname, ndmin=2).shape[1] == ncol

    drs = np.loadtxt(out / "drs.dat", ndmin=2)
    np.testing.assert_allclose(drs[:, 1], solver.discrete.tau, rtol=1e-6)


def test_save_default_directory(solver, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    solver.save()
    assert {p.name for p in tmp_path.iterdir()} == BASE_FILES


def test_save_records_G0_when_plateau(tmp_path):
    s = ReSpect(ReSpectConfig(plateau=True)).fit(TESTS / "test7.dat")
    s.save(path=tmp_path)
    for fname, G0 in (("crs.dat", s.continuous.G0), ("drs.dat", s.discrete.G0)):
        first = headers(tmp_path / fname)[0]
        assert first.startswith("# G0 = ")
        assert float(first.split("=")[1]) == pytest.approx(G0, rel=1e-6)


def test_invalid_which(solver):
    for method in (solver.save, solver.plot):
        with pytest.raises(ReSpectError, match="Invalid 'which'"):
            method(which="everything")


def test_plot_returns_figures(solver, tmp_path):
    from matplotlib.figure import Figure

    figs = solver.plot(which="base", toFile=True, path=tmp_path)
    assert len(figs) == 1 and isinstance(figs[0], Figure)
    assert (tmp_path / "Gfit.pdf").exists()

    figs = solver.plot(which="full", toFile=True, path=tmp_path)
    assert len(figs) == 2 and all(isinstance(f, Figure) for f in figs)
    assert (tmp_path / "diagnostics.pdf").exists()

    # returned figures can still be customized and saved again
    figs[0].axes[0].set_title("custom")
    figs[0].savefig(tmp_path / "custom.pdf")
    assert (tmp_path / "custom.pdf").exists()


def test_fit_and_save_chain(tmp_path):
    s = ReSpect().fit(TESTS / DATA).save(which="base", path=tmp_path)
    assert isinstance(s, ReSpect)


# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------

def test_example_toml_matches_defaults():
    assert ReSpect.from_toml(ROOT / "inp.toml").config == ReSpectConfig()


def test_toml_overrides(tmp_path):
    f = tmp_path / "inp.toml"
    f.write_text("[spectrum]\nns = 60\nplateau = true\n[io]\nresample = false\n")
    cfg = ReSpectConfig.from_toml(f)
    assert (cfg.ns, cfg.plateau, cfg.resample) == (60, True, False)


# ---------------------------------------------------------------------------
# Names kept for backward compatibility (package-specific)
# ---------------------------------------------------------------------------

def test_deprecated_save_argument_order(solver, tmp_path):
    with pytest.warns(DeprecationWarning, match="save\\(which, path\\)"):
        solver.save(str(tmp_path / "a"))
    assert {p.name for p in (tmp_path / "a").iterdir()} == BASE_FILES

    with pytest.warns(DeprecationWarning):
        solver.save(str(tmp_path / "b"), "full")
    assert {p.name for p in (tmp_path / "b").iterdir()} == FULL_FILES


def test_deprecated_N_opt(solver):
    d = solver.discrete
    with pytest.warns(DeprecationWarning, match="N_opt"):
        assert d.N_opt == d.N_bst[np.argmin(d.AIC_bst)]
