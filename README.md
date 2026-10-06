# pyReSpect-time-2.0

A rewrite of the classic python library for extracting continuous and discrete relaxation spectra from stress relaxation data $G(t)$. The core algorithms are the same. The interface (both developer and user) is modernized.

**If you need the legacy codebase, it is archived at [github.com/shane5ul/pyReSpect-time-legacy](https://github.com/shane5ul/pyReSpect-time-legacy).**

> **Sister package.** [pyReSpect-freq](https://github.com/shane5ul/pyReSpect-freq) does the same job for frequency-domain data $G^*(\omega)$. The two packages share one interface: the same `ReSpect` class, configuration, result fields and output files. Only the data passed to `fit()` differ.

pyReSpect solves the regularized inverse problem

$$G(t) = G_0 + \int_{-\infty}^{\infty} e^{H(s)}\, e^{-t/s}\ d\ln s$$

to recover the continuous relaxation spectrum (CRS) $h(s) = e^{H(s)}$, and subsequently fits a discrete Maxwell model (DRS)

$$G(t) = G_0 + \sum_{i=1}^{N} g_i e^{-t/\tau_i}$$

where $N$ is selected via an information criterion.

## References

- Shanbhag, S., "pyReSpect: A Computer Program to Extract Discrete and Continuous Spectra from Stress Relaxation Experiments", *Macromolecular Theory and Simulations*, **2019**, 1900005. [doi:10.1002/mats.201900005](https://doi.org/10.1002/mats.201900005)
- Takeh, A. and Shanbhag, S., "A computer program to extract the continuous and discrete relaxation spectra from dynamic viscoelastic measurements", *Applied Rheology*, **2013**, 23, 24628.

---

## Features

- Easy installation
- Library functions can be imported and called from other programs
- Clean object-oriented API with method chaining
- Simplified user-interface:
  - configuration (old `inp.dat`) and data (old `Gt.dat`) can be supplied both programmatically or via files
  - TOML configuration file support
- It separates computation from I/O and plotting.
- Continuous spectrum via Tikhonov regularization with Bayesian $\lambda$ selection
- Discrete Maxwell modes via AIC minimization and NLLS fine-tuning
- Optional plateau modulus $G_0$ for viscoelastic solids

---

## Installation

### Requirements

- Python >= 3.12
- numpy >= 2.4
- scipy >= 1.17
- matplotlib

### From GitHub

```bash
pip install git+https://github.com/shane5ul/pyReSpect-time.git
```

### For development

Clone the repository and install in editable mode from the repo root:

```bash
git clone https://github.com/shane5ul/pyReSpect-time.git
cd pyReSpect-time
pip install -e ".[test]"
pytest
```

---

## Quick start

```python
from pyrespect_time import ReSpect

solver = ReSpect()
solver.fit("tests/test2.dat")       # or solver.fit(t, Gt)

# Access results
print(solver.continuous.H)    # log of the continuous spectrum, H(s)
print(solver.discrete.tau)    # discrete relaxation times
print(solver.discrete.g)      # discrete mode weights

# Save and plot
solver.save(which="full", path="output/")
figs = solver.plot(which="base")
```

`fit()` and `save()` return the solver, so they can be chained; `plot()` returns the figures:

```python
ReSpect().fit("Gt.dat").save(which="base", path="output/")
```

The same script is in `quickstart.py`.

---

## Input data

`fit()` accepts a **file**, **arrays**, or a **tuple of arrays**.

### File format

A plain-text file with whitespace-separated columns. Two formats are supported:

| Columns | Meaning | Resampled? |
|---------|---------|------------|
| 2: `t  G(t)` | raw stress relaxation data | yes (onto a 100-point geometric grid) |
| 3: `t  G(t)  weight` | pre-processed data with per-point weights | no |

Duplicate time values are removed and the data are sorted automatically. Lines beginning with `#` are ignored.

### Array input

```python
import numpy as np

t  = np.logspace(-2, 2, 50)
Gt = ...   # G(t)

solver.fit(t, Gt)                    # separate arrays
solver.fit((t, Gt))                  # the same, as a tuple
solver.fit(t, Gt, weights=wt)        # with per-point weights
solver.fit((t, Gt, wt))              # the same, as a tuple
solver.fit(np.loadtxt("Gt.dat"))     # 2-D array with the columns of the file
```

### Weights and resampling

`weights` has one entry per data point. Data supplied with weights, in any of the forms above, are treated as pre-processed and are used as-is.

Data without weights are resampled onto a geometric grid of `n_resample` points. Switch this off with `ReSpectConfig(resample=False)`, or for a single call with `solver.fit(..., resample=False)`.

After `fit()`, the data actually used are available as `solver.t`, `solver.Gt` and `solver.weights`.

---

## Configuration

Settings are passed via a `ReSpectConfig` object. Every parameter has a sensible default, so `ReSpectConfig()` works out of the box. Set `plateau = True` if you are modeling a viscoelastic solid with non-zero $G_0$.

```python
from pyrespect_time import ReSpect, ReSpectConfig

config = ReSpectConfig(
    ns       = 100,       # CRS grid points
    plateau  = False,     # set True for viscoelastic solids (fits G0)
    freq_end = "lenient", # s-axis window: "lenient" | "neutral" | "strict"
)
solver = ReSpect(config).fit("Gt.dat")
```

### Configuration from file

Configuration can also be loaded from a TOML file. Only the settings you want to override need to be listed; `inp.toml` in the repo root lists them all with their defaults.

```toml
[spectrum]
ns       = 200
plateau  = true

[discrete]
max_num_modes = 5

[io]
n_resample = 50
```

```python
solver = ReSpect.from_toml("inp.toml").fit("Gt.dat")
```

`ReSpect.from_yaml()` does the same for YAML files (requires `pyyaml`).

### Full parameter reference

#### Continuous spectrum

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ns` | int | 100 | Number of grid points for $s$ (the CRS axis). Typical range: 50–200. |
| `plateau` | bool | False | Fit a non-zero plateau modulus $G_0$ (for viscoelastic solids). |
| `freq_end` | str | `"lenient"` | How far $s$ extends beyond the time window. |

#### Regularization ($\lambda$ selection)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lam_min` | float | 1e-10 | Lower bound of the $\lambda$ search grid. |
| `lam_max` | float | 1e3 | Upper bound of the $\lambda$ search grid. |
| `lam_C` | float \| None | None | Pin $\lambda$ to this value instead of using the Bayesian L-curve. When set, `dH` (the error band) is not computed. |
| `lam_density` | int | 2 | $\lambda$ grid points per decade. Increase for a finer search. |
| `SmFacLam` | float | 0.0 | Smoothness nudge in $[-1, 1]$. Positive values push $\lambda$ toward `lam_max` (smoother spectrum); negative values push toward `lam_min` (rougher). |

#### Discrete spectrum

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_num_modes` | int \| None | None | Cap on the number of Maxwell modes scanned. `None` = auto. |
| `delta_base_weight_dist` | float | 0.2 | Step size for the AIC scan over the base weight parameter $w_b \in (0,1)$. Smaller → finer scan, higher cost. |
| `min_tau_spacing` | float | 1.25 | Minimum allowed ratio $\tau_{i+1}/\tau_i$. Pairs closer than this are merged. Must be > 1. |

#### Input / output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resample` | bool | True | Resample input data onto a geometric grid. No effect on data supplied with weights. |
| `n_resample` | int | 100 | Number of points in the resampled grid. |

---

## Results

After `fit()`, the results are in `solver.continuous` and `solver.discrete`.

| `solver.continuous` | Shape | Description |
|---------------------|-------|-------------|
| `s` | (ns,) | Relaxation time grid. |
| `H` | (ns,) | Log-CRS $H(s)$; the spectrum is $h(s) = e^{H(s)}$. |
| `dH` | (ns,) | Error band on $H$. `None` if `lam_C` was pre-specified. |
| `G_fit` | (n,) | predicted $G(t)$, from the CRS. |
| `G0` | float | Plateau modulus (0 if `plateau=False`). |
| `lam_C` | float | The $\lambda$ used. |
| `lam`, `rho`, `eta`, `log_P`, `H_lam` | | L-curve scan: $\lambda$ grid, $\rho(\lambda)$, $\eta(\lambda)$, $\log p(\lambda)$, and $H$ at each $\lambda$. `None` if `lam_C` was pre-specified. |

| `solver.discrete` | Shape | Description |
|-------------------|-------|-------------|
| `g` | (N,) | Mode weights $g_i$. |
| `tau` | (N,) | Relaxation times $\tau_i$, increasing. |
| `dtau` | (N,) | Uncertainty of $\tau_i$ (`nan` if the NLLS fine-tuning was not used). |
| `N` | int | Number of modes. |
| `G_fit` | (n,) | predicted $G(t)$, from the DRS. |
| `G0` | float | Plateau modulus (0 if `plateau=False`). |
| `error` | float | Weighted sum of squared relative residuals of the discrete fit. |
| `wt_base`, `AIC_bst`, `N_bst`, `nz_N_bst` | | AIC scan: base weights $w_b$, best AIC, and the number of modes requested / surviving at each $w_b$. |

---

## Output files

`solver.save(which, path)` writes results to `path/`. Two levels of output are available. Every file starts with a `#` header line naming its columns.

### `which="base"` — main results

| File | Columns | Description |
|------|---------|-------------|
| `crs.dat` | `s  h` | Continuous spectrum $h(s) = e^{H(s)}$. A first header line stores $G_0$ when `plateau=True`. |
| `drs.dat` | `g  tau  dtau` | Discrete Maxwell modes. A first header line stores $G_0$ when `plateau=True`. |
| `Gfit.dat` | `t  G_cont  G_disc` | Model fits from both CRS and DRS vs time. |

### `which="full"` — diagnostics (in addition to base)

| File | Columns | Description |
|------|---------|-------------|
| `logPlam.dat` | `lambda  logP` | Log Bayesian evidence $\log p(\lambda)$ vs $\lambda$. Peak locates $\lambda_M$. |
| `rho-eta.dat` | `lambda  rho  eta` | L-curve data: $\rho(\lambda)$ (data misfit) and $\eta(\lambda)$ (curvature penalty). |
| `aic.dat` | `wt_base  N_bst  AIC  nz_N_bst` | AIC scan: for each $w_b$, the number of modes requested at the best AIC, the AIC value, and the number of modes that survive pruning. |

`logPlam.dat` and `rho-eta.dat` are skipped when `lam_C` is pre-specified (the L-curve is not computed in that case).

---

## Plotting

```python
figs = solver.plot(which="base")                         # interactive display
figs = solver.plot(which="full", toFile=True, path="output/")  # save PDFs
```

`plot()` returns the list of matplotlib figures, so they can be customized further.

`which="base"` produces a two-panel figure (`Gfit.pdf`):

- **Left**: $h(s) = e^{H(s)}$ with $\pm 2.5\,\Delta H$ error band, overlaid with
  discrete mode weights $g_i$ vs $\tau_i$.
- **Right**: experimental $G(t)$ data against continuous and discrete fits
  on a log-log axis.

`which="full"` adds a three-panel diagnostic figure (`diagnostics.pdf`): $\log p(\lambda)$ vs
$\lambda$, the $\rho$-$\eta$ L-curve with the chosen $\lambda$ marked, and the AIC scan.

---

## Package layout

The package is structured so that **all scientific computation is separated from I/O and plotting**. `continuous.py`, `discrete.py`, and `kernels.py` are pure numerical modules; they read no files and produce no figures. This makes them straightforward to call from other programs or notebooks. The modules, functions and result fields have the same names in pyReSpect-freq.

---

## Changes in version 2.1

Version 2.1 aligns the interface with pyReSpect-freq.

- **`save` argument order.** The signature is now `save(which="base", path="./")`, as in pyReSpect-freq. The old positional call `save("output/")` still works but emits a `DeprecationWarning`; calls with keywords are unaffected.
- **`fit` input.** Besides a file name or `fit(t, Gt)`, `fit` now accepts a `pathlib.Path`, a list or tuple of arrays (optionally with weights as a third entry), and a 2-D array.
- **Resampling.** `config.resample` is now honoured (it was ignored); `fit(..., resample=...)` overrides it for one call.
- **Weights.** `fit(t, Gt, weights=w)` used to resample the data and silently reset the weights to 1. Data supplied with weights are now used as-is.
- **Result objects.** `G0` is `0.0` instead of `None` when `plateau=False`. New fields: `continuous.dH` (error band on $H$), `discrete.N` (number of modes returned), `discrete.error` and `discrete.nz_N_bst`. `discrete.N_opt` is deprecated: it was the number of modes *requested* at the AIC optimum, not the number returned. Construct the result dataclasses by keyword; the field order changed.
- **Discrete spectrum.** As in pyReSpect-freq, the fine-tuned $\tau_i$ are confined to $0.02\,t_{\min} \le \tau_i \le 50\,t_{\max}$ and the fine-tuning is kept only if it improves the fit. On the test data this moves $\tau_i$ by less than 2 parts in $10^5$ and leaves the number of modes unchanged. `dtau` is now reordered together with `tau`.
- **Output files.** Every file has a header naming its columns. `aic.dat` has a fourth column, `nz_N_bst`.
- **Plots.** The spectrum panel shows the $\pm 2.5\,\Delta H$ error band; labels match pyReSpect-freq. Figures saved with `toFile=True` are closed after saving.
- **Warnings.** A `ReSpectWarning` is issued when only one value of $N$ is scanned.
- **Exports.** `ReSpectWarning`, `ContinuousResult`, `DiscreteResult` and `__version__` are available from the package.

---

## Citation

If you use pyReSpect in your research, please cite:

```bibtex
@article{shanbhag2019pyrespect,
  author  = {Shanbhag, Sachin},
  title   = {pyReSpect: A Computer Program to Extract Discrete and Continuous
             Spectra from Stress Relaxation Experiments},
  journal = {Macromolecular Theory and Simulations},
  year    = {2019},
  volume  = {28},
  pages   = {1900005},
  doi     = {10.1002/mats.201900005}
}

@article{takeh2013computer,
  author  = {Takeh, Arsia and Shanbhag, Sachin},
  title   = {A computer program to extract the continuous and discrete
             relaxation spectra from dynamic viscoelastic measurements},
  journal = {Applied Rheology},
  year    = {2013},
  volume  = {23},
  pages   = {24628}
}
```

---

## History and acknowledgements

Development was supported by National Science Foundation DMR grants 0953002 and 1727870. The code is based on the MATLAB program [ReSpect](https://www.mathworks.com/matlabcentral/fileexchange/54322-respect-v2-0).
