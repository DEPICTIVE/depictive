
DEPICTIVE : DEtermining Parameter Influence on Cell To cell Variability Through the Inference of Variance Explained
===========================================

This package provides simple tools for dissecting sources of cell-to-cell variability from single cell data and perturbation experiments.  In addition we provide tools for generating simulation data for testing inference strategy and plotting.

For and example script that demonstrates these tools please refer to the `examples/cli_run_example.py`.

Installation
------------

DEPICTIVE is not yet distributed through the Python Package Index (PyPI).  To install it on your system you will need to clone this repo, install dependencies and then install the package.

### Dependencies

To use this package you will need:

- Python (3.6)
- Numpy (>= 1.12.0)
- SciPy (>= 0.18.1)
- matplotlib (>= 2.0.0)

These packages can be locally installed using `pip` and the `requirements.txt`.  First, ensure pip is up to date by executing the following on the command line,

```bash
$ pip install --upgrade pip
```

and then move in to the clone directory, and

```bash
$ pip install -r requirements.txt
```

### Install

Lastly, to install the development version of DEPICTIVE,

```bash
$ pip install [path into outermost depictive directory]
```

Using the package
-----------------

Check out our example located at `examples/cli_run_example.py`.

Citation
--------

Please don't forget to check out our paper on [bioRxiv](https://www.biorxiv.org/content/early/2017/10/10/201160) and reference us
```
@article {2017mito_sims,
  author = {Santos, L.C. and Vogel, R. and Chipuk, J.E. and
    Birtwistle, M.R. and Stolovitzky, G. and Meyer, P.},
  title = {Origins of fractional control in regulated cell death},
  year = {2017},
  doi = {10.1101/201160},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2017/10/10/201160},
  eprint = {https://www.biorxiv.org/content/early/2017/10/10/201160.full.pdf},
  journal = {bioRxiv}
}
```
