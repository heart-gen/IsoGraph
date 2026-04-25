# Installation

## PyPI (recommended)

Install the core package from PyPI:

```bash
pip install isograph
```

Supports Python `3.11` through `3.14`.

## Optional Backends

### VAE and GPU-latent

The `vae` and `gpu_latent` backends require PyTorch. Install a build appropriate for
your environment before using `backend=vae`, `backend=gpu_latent`, or importing
`isograph.models.vae` / `isograph.models.gpu_latent`:

```bash
pip install torch
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for
GPU/CUDA builds.

### WGCNA

The `wgcna` backend requires R with the `WGCNA` package and the `rpy2` Python binding:

```bash
pip install rpy2
```

In R:

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
