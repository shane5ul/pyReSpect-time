# pyReSpect-time-2.0

A rewrite of the python classic library for extracting continuous and discrete relaxation spectra from stress relaxation data $G(t)$. The core algorithms are the same, the interface (both developer and user) is modernized.

**If you need the legacy codebase has been archived at github.com/shane5ul/pyReSpect-time-legacy.**

The method solves the regularized inverse problem

$$G(t) = G_0 + \int_{-\infty}^{\infty} H(s) e^{-t/s}\ d\ln s$$

to recover the continuous relaxation spectrum $H(s)$, and subsequently fits a discrete Maxwell model

$$G(t) = G_0 + \sum_{i=1}^{N} g_i e^{-t/\tau_i}$$

where $N$ is selected via an information criterion.

The papers describing the core algorithm and test cases are:

> Shanbhag, S., "pyReSpect: A Computer Program to Extract Discrete and Continuous Spectra from Stress Relaxation Experiments", *Macromolecular Theory and Simulations*, **2019**, 1900005. [doi:10.1002/mats.201900005](https://doi.org/10.1002/mats.201900005)

> Takeh, A. and Shanbhag, S., "A computer program to extract the continuous and discrete relaxation spectra from dynamic viscoelastic measurements", *Applied Rheology*, **2013**, 23, 24628.

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
cd pyrespect
pip install -e .
```

---

## Quick start

```python
from pyrespect import ReSpect, ReSpectConfig

# Default settings — fit from a data file
solver = ReSpect()
solver.fit("Gt.dat")  # "Gt.dat" file contains data

# Access results
print(solver.continuous.H)    # continuous spectrum H(s)
print(solver.discrete.tau)    # discrete relaxation times
print(solver.discrete.g)      # discrete mode weights

# Save and plot
solver.save(which="base", path="output/")
solver.plot(which="base", toFile=True, path="output/")
```

Method chaining is supported:

```python
ReSpect().fit("Gt.dat").save(which="base", path="output/")
```

---

## Input data format

The input file should contain two or three whitespace-delimited columns:

| Column | Description |
|--------|-------------|
| 1 | Time $t$ |
| 2 | Relaxation modulus $G(t)$ |
| 3 | Per-datapoint weight $w$ (optional) |

When only two columns are supplied, the data is resampled onto a geometric grid of 100 points and uniform weights are used. When three columns are supplied, the data is assumed to be pre-processed and is used as-is.

---

## Configuration

Settings are passed via a `ReSpectConfig` object. All parameters have sensible defaults. Set `plateau = True` if you are modeling a viscoelastic solid with non-zero $G_0$.

```python
config = ReSpectConfig(
    ns=100,               # discretization points for s axis
    plateau=False,        # infer plateau modulus G0?
    freq_end="lenient",   # s-axis extent: "lenient", "neutral", or "strict"
    lam_min=1e-10,        # lower bound for lambda search
    lam_max=1e3,          # upper bound for lambda search
    lam_C=None,           # pre-specify lambda (None = auto via L-curve)
    SmFacLam=0.0,         # smoothness nudge in [-1, 1]
    max_num_modes=None,   # cap on discrete modes
    delta_base_weight_dist=0.2,  # AIC scan step size
    min_tau_spacing=1.25, # minimum tau_{i+1}/tau_i before merging
)
solver = ReSpect(config)
```

Configuration can also be loaded from a TOML file (only required when you want to override defaults):

```python
solver = ReSpect.from_toml("inp.toml")
```

Example `inp.toml`:

```toml
[spectrum]
ns = 200
plateau = true

[regularization]
SmFacLam = 0.5

[discrete]
max_num_modes = 5

[io]
n_resample = 50
```

---

## API reference

### `ReSpect`

The primary solver class.

| Method | Description |
|--------|-------------|
| `ReSpect(config)` | Construct solver with optional `ReSpectConfig` |
| `fit(source, ...)` | Fit continuous and discrete spectra |
| `save(path, which)` | Write results to files |
| `plot(path, which, ...)` | Plot spectra and diagnostics |

### `ReSpectConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ns` | int | 100 | Discretization points for $s$ axis |
| `plateau` | bool | False | Infer plateau modulus $G_0$ |
| `freq_end` | str | `"lenient"` | $s$-axis extent relative to $t$ window |
| `lam_min` | float | 1e-10 | Lower bound for $\lambda$ search |
| `lam_max` | float | 1e3 | Upper bound for $\lambda$ search |
| `lam_C` | float\|None | None | Pre-specified $\lambda$ (skips L-curve) |
| `SmFacLam` | float | 0.0 | Smoothness nudge in $[-1, 1]$ |
| `max_num_modes` | int\|None | None | Cap on number of discrete modes |
| `delta_base_weight_dist` | float | 0.2 | AIC scan step size |
| `min_tau_spacing` | float | 1.25 | Minimum $\tau_{i+1}/\tau_i$ before merging |
| `resample` | bool | True | Resample input data? |
| `n_resample` | int | 100 | Points for geometric resampling |

### Output files (`save`)

| File | Contents |
|------|----------|
| `crs.dat` | Continuous spectrum: $s$, $\exp(H(s))$ |
| `drs.dat` | Discrete modes: $g_i$, $\tau_i$, $\delta\tau_i$ |
| `Gfit.dat` | Model fits: $t$, $G_\text{cont}(t)$, $G_\text{disc}(t)$ |
| `rho-eta.dat` | L-curve data: $\lambda$, $\rho$, $\eta$ (`which="full"` only) |
| `logPlam.dat` | Bayesian evidence: $\lambda$, $\log p(\lambda)$ (`which="full"` only) |
| `aic.dat` | AIC scan: $w_b$, $N_\text{bst}$, AIC (`which="full"` only) |

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

## Acknowledgements

Development was supported by National Science Foundation DMR grants 0953002 and 1727870. The code is based on the MATLAB program [ReSpect](
https://www.mathworks.com/matlabcentral/fileexchange/54322-respect-v2-0)
