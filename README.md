# IsoGraph

IsoGraph is a Python research software package for discovering isoform-switch and splicing
modules from bulk RNA-seq. It combines gene-local compositional modeling with network
inference so researchers can move from transcript-level counts to gene-module structure,
trait associations, and reproducible benchmark artifacts.

## Core Capabilities

- Generate and benchmark against the permanent `core_v1` fixture suite and the large-scale
  `scale_v1` suite (6k–12k genes, 25:1–50:1 genes-to-samples ratios).
- Freeze the bundled `real_caudate_aa_v1` real-data fixture from local BrainSeq inputs.
- Fit any backend on a prepared dataset bundle via `isograph fit` (VAE default).
- Run `baseline`, `latent`, `graph`, `vae`, or `wgcna` backends programmatically or
  through the benchmark runner.
- Export reproducible artifacts, benchmark reports, calibration summaries, and snapshot comparisons.
- Explain discovered modules at transcript-feature resolution using `isograph explain-module`:
  gene drivers, transcript polarity, high-vs-low contrasts, publication-ready plots, optional
  VAE decoder attribution, and Captum Integrated Gradients encoder attribution.
- Annotate transcript switch pairs with GTF-derived structural labels (exon changes, CDS/UTR
  shifts, biotype switches) using `isograph annotate-structure`.

## Installation

Install the core package from PyPI:

```bash
pip install isograph
```

The core package supports Python `3.11` through `3.14`.

### Optional backends

IsoGraph installs `mpmath`, which is required by modern SymPy releases. The
`vae` backend also requires PyTorch, but PyTorch is intentionally not installed
by IsoGraph because CPU/GPU/CUDA builds are platform-specific. Install the build
that matches your system before using it:

```bash
pip install torch
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for
GPU/CUDA builds.

The `wgcna` backend requires R with the `WGCNA` package and `Rscript` on `PATH`.

## Quickstart

Run a minimal benchmark on the bundled toy fixture (VAE is the default backend):

```bash
isograph benchmark -- \
  fixture_filter=toy_v1 \
  stage_name=readme_smoke
```

This writes benchmark artifacts under `artifacts/benchmarks/readme_smoke/toy_v1/` and
JSON reports under `artifacts/reports/`.

## Using Your Own Data

IsoGraph expects a prepared dataset bundle containing a `manifest.json`, aligned sample
metadata, feature tables, and dense count matrices. The `fit` command supports all
backends; VAE is the default:

```bash
isograph fit \
  --dataset-path path/to/my_dataset_bundle \
  --output-dir artifacts/fits/my_dataset
```

To switch backends or pass Hydra overrides:

```bash
isograph fit \
  --dataset-path path/to/my_dataset_bundle \
  --backend baseline \
  --output-dir artifacts/fits/my_dataset_baseline

isograph fit \
  --dataset-path path/to/my_dataset_bundle \
  --backend vae \
  --output-dir artifacts/fits/my_dataset_vae \
  -- vae.hidden_dim=256 vae.n_epochs=400
```

After fitting, explain one or more modules:

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --module-ids M000 M001 \
  --plot \
  --output-dir artifacts/explain/my_dataset
```

The detailed walkthroughs live in the Wiki, and the formal data model is documented
in the RTD source tree.

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
- The VAE backend requires a separate PyTorch installation.
- The WGCNA backend requires R with the `WGCNA` package installed.
- The `freeze-real` workflow depends on local BrainSeq-style source files and is not a
  generic data-ingestion command for arbitrary cohorts.
- VAE decoder attribution (`--vae-attribution`) and Captum Integrated Gradients
  (`--integrated-gradients`) require a VAE checkpoint in the fit artifact directory and,
  for Integrated Gradients, `pip install isograph[torch-explain]`.
