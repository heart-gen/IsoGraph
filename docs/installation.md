# Installation

## Conda Environment

The canonical local setup uses the repository environment file:

```bash
conda env create -f environment.yml
conda activate isograph
isograph --help
```

This installs the package in editable mode with development dependencies.

## Minimal Editable Install

If you manage Python yourself, a direct editable install is also supported:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Documentation Dependencies

To build the Sphinx documentation locally:

```bash
python -m pip install -e .[docs]
python -m sphinx -W -b html docs docs/_build/html
```

## Optional VAE Dependency

The VAE backend requires PyTorch in addition to the package dependencies shipped in
`pyproject.toml`. Install a PyTorch build appropriate for your environment before using
`backend=vae` or importing `isograph.models.vae`.
