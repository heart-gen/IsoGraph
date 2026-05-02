# Installation

## PyPI (recommended)

Install the core package from PyPI:

```bash
pip install isograph
```

Supports Python `3.11` through `3.14`.

## Optional Backends

### VAE

IsoGraph installs `mpmath`, which is required by modern SymPy releases. The
`vae` backend also requires PyTorch, but PyTorch is intentionally not installed
by IsoGraph because CPU/GPU/CUDA builds are platform-specific. Install the build
that matches your environment before using `backend=vae` or importing
`isograph.models.vae`:

```bash
pip install torch
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for
GPU/CUDA builds.

### WGCNA

The `wgcna` backend requires R with the `WGCNA` package and `Rscript` on `PATH`:

```r
install.packages("WGCNA")
```

## Documentation

To build the Sphinx documentation locally:

```bash
pip install isograph[docs]
python -m sphinx -W -b html docs docs/_build/html
```

## Development

To install in editable mode with development dependencies:

```bash
git clone https://github.com/heart-gen/IsoGraph
cd IsoGraph
pip install -e .[dev]
```

Install PyTorch separately when developing or testing the VAE backend.
