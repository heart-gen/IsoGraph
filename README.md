# IsoGraph

IsoGraph is a Python research software package for discovering isoform-switch and splicing
modules from bulk RNA-seq. It combines gene-local compositional modeling with network
inference so researchers can move from transcript-level counts to gene-module structure,
trait associations, and reproducible benchmark artifacts.

## Status

IsoGraph currently includes completed development stages 0 through 4:

- Stage 0: package, CLI, config validation, fixtures, and reproducibility infrastructure
- Stage 1: deterministic baseline network backend
- Stage 2: latent probabilistic backend with stability selection
- Stage 3: graph-aware backend
- Stage 4: VAE backend

Stage 5, a WGCNA comparison benchmark on simulated data, is planned next.

## Core Capabilities

- Generate and benchmark against the permanent `core_v1` fixture suite.
- Freeze the bundled `real_caudate_aa_v1` real-data fixture from local BrainSeq inputs.
- Fit the deterministic baseline backend from the command line on a prepared dataset bundle.
- Run baseline, latent, graph, or VAE backends programmatically, or through the benchmark runner.
- Export reproducible artifacts, benchmark reports, calibration summaries, and snapshot comparisons.

## Installation

The repository ships with a conda environment that installs IsoGraph in editable mode:

```bash
conda env create -f environment.yml
conda activate isograph
isograph --help
```

If `conda` is not initialized in the current shell, run `eval "$(conda shell.bash hook)"`
first or initialize conda for your shell.

The core package supports Python `3.11` through `3.14`. The bundled environment uses
Python `3.11` as the canonical local development runtime.

## Quickstart

Run a minimal benchmark on the bundled toy fixture:

```bash
conda activate isograph
isograph benchmark -- \
  backend=baseline \
  fixture_filter=toy_v1 \
  stage_name=readme_smoke
```

This writes benchmark artifacts under `artifacts/benchmarks/readme_smoke/toy_v1/` and
JSON reports under `artifacts/reports/`.

## Using Your Own Data

IsoGraph expects a prepared dataset bundle containing a `manifest.json`, aligned sample
metadata, feature tables, and dense count matrices. The current command-line path for
custom data is:

```bash
isograph fit \
  --dataset-path path/to/my_dataset_bundle \
  --output-dir artifacts/fits/my_dataset
```

At present, `fit` runs the deterministic baseline backend. For latent, graph, or VAE
backends on your own bundle, use the Python API directly. The detailed walkthroughs live
in the Wiki, and the formal data model is documented in the RTD source tree.

## Documentation

- Reference docs for publication on Read the Docs live in [docs](docs/index.md).
- Step-by-step tutorials for installation, data preparation, and own-data workflows live
  in the [GitHub Wiki](https://github.com/heart-gen/IsoGraph/wiki).
- Project planning and staged development history remain in
  [docs/staged-roadmap.md](docs/staged-roadmap.md).

## Citation

If you use IsoGraph in research, cite the software repository using the metadata in
[CITATION.cff](CITATION.cff). If a manuscript or preprint becomes available later, that
can be added as a preferred citation target without changing the software citation path.

## Acknowledgements

IsoGraph is supported by the National Institute on Minority Health and Health Disparities
award `R00 MD0169640` and the Alzheimer's Association award `25AARG-1413315`.

## Reproducibility and Data Provenance

- The benchmark suite is fixture-driven and designed to preserve regression targets across
  development stages.
- The bundled real-data workflow freezes a reproducible `real_caudate_aa_v1` dataset from
  local BrainSeq-derived inputs and caches intermediate selections under
  `benchmarks/cache/real_data/`.
- Benchmark, calibration, runtime, and snapshot artifacts are written into versioned
  directories under `artifacts/` and `snapshots/`.

## Limitations

- The benchmark CLI is optimized for the bundled fixture suite rather than arbitrary
  user-defined suites.
- The `fit` CLI currently exposes only the baseline backend for custom datasets.
- The VAE backend requires a separate PyTorch installation.
- The bundled `freeze-real` workflow depends on local BrainSeq-style source files and is
  not a generic data-ingestion command for arbitrary cohorts.
