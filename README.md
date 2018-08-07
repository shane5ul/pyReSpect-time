# pyReSpect-time

Extract continuous and discrete relaxation spectra from G(t)

## Files

### Code Files

This repository contains two python modules `contSpec.py` `discSpec.py`. They extract the continuous and discrete relaxation spectra from a stress relaxation data. (t versus G(t) experiment or simulation).

It containts a third module `common.py` which contains utilities required by both `contSpec.py` and `discSpec.py`.

In addition to the python modules, jupyter notebooks `interactContSpec.ipynb` and `interactDiscSpec.ipynb` are also included. These allow the user to experiment with parameter settings interactively.

### Input Files

The user is expected to supply two files:

+ `inp.dat` is used to control parameters and settings
+ `Gt.dat` which contains two columns of data `t` and `G(t)`

### Output Files

Text files containting output from the code are stored in a directory `output/`. These include a fit of the data, the spectra, and other files relevant to the continuous or discrete spectra calculation. 

Graphical and onscreen output can be suppressed by appropriate flags in `inp.dat`.

## Usage

Once `inp.dat` and `Gt.dat` are furnished, running the code is simple.

To get the continuous spectrum:

`python3 contSpec.py`

The **continuous spectrum must be extracted before the discrete spectrum** is computed. The discrete spectrum can then be calculated by

`python3 discSpec.py`

### Interactive Mode

The interactive mode offers a "GUI" for exploring parameter settings. To launch use `jupyter notebook interactContSpec.ipynb` or `jupyter notebook interactContSpec.ipynb`.

### Pre-requisites

The numbers in parenthesis show the version this has been tested on. 

python3 (3.5.2)
numpy (1.14.2)
scipy (1.0.1)

For interactive mode:

jupyter (4.3)
ipywidgets (6.0.0)

## History

The code is based on the Matlab program [ReSpect](https://www.mathworks.com/matlabcentral/fileexchange/40458-respect), which extract the continuous and discrete relaxation spectra from frequency data, G*(w).

### Major Upgrade: August 2008

+ orignal program with n = 100 and lambda = 20 pts with clean 1 mode data took ~33s.
+ prestore kernMat: evaluation of kernel (meshgrid S, T, and hs) by prestoring kernMat ~12.5s (~3x speed gain)
+ improved least_squares setting by incorporating jacobianLM; ~6s (2x gain)
+ lcurve coarser (auto) mesh, robust criterion, and interpolation ~3.5s (~1.75x gain)
+ Total gain in speed as a consequence of these improvements is 33s -> 3.5s a nearly 10x gain!

