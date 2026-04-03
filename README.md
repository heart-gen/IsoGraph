# IsoGraph

IsoGraph is a benchmark-first toolkit for gene-aware compositional latent network modeling.
The repository is intentionally organized so benchmark infrastructure and a deterministic
reference baseline exist before later graph-aware or variational models are added.

## Design Rules

- Every model must run on the permanent `core_v1` benchmark suite:
  `toy_v1`, `medium_v1`, and `real_caudate_aa_v1`.
- Runtime code is split into five layers: `io`, `features`, `models`,
  `evaluation`, and `workflow`.
- The stage-1 baseline stays in the repository as a regression target even if
  later models outperform it.
- Biological sophistication and software-complexity growth should not land in
  the same commit.

## Project Layout

```text
src/isograph/
  io/          dataset manifests, artifacts, real-data freeze pipeline
  features/    residualization and gene-aware feature transforms
  models/      deterministic baseline and future model interfaces
  evaluation/  benchmark runners and metrics
  workflow/    CLI entry points and typed configs
configs/       Hydra configs
docs/          staged implementation tracker
tests/         unit and property-based tests
```

## Environment

Create a dedicated conda environment for this repository:

```bash
conda env create -f environment.yml
conda activate isograph
```

The environment file installs the project in editable mode with dev dependencies.

## Quickstart

Install and validate a fresh checkout:

```bash
conda env create -f environment.yml
conda activate isograph
isograph --help
pytest -q tests/test_synthetic.py
```

The package is intended to work on Python `3.11` through `3.14`. The conda
environment file pins `3.11` as the canonical local development runtime, while
CI should validate the wider supported range.

If `conda` is not initialized in the current shell, use
`eval "$(conda shell.bash hook)"` first or initialize it according to your
local installation.

## CLI

The package exposes a single CLI with benchmark-first entry points:

```bash
isograph freeze-real
isograph benchmark
isograph fit
isograph compare
isograph export
```

Each command accepts Hydra-style overrides after `--`, for example:

```bash
isograph benchmark -- dataset_suite=core_v1 model.alpha=0.12
```

`freeze-real` uses a repo-local cache by default at
`benchmarks/cache/real_data`. The first population pass may need to read the
upstream BrainSeq sources, but later Stage 0 smoke runs should reuse the local
cached sample selection, projected gene counts, partitioned transcript-count
cache, and frozen `realmini` fixture.

## Status

Implementation progress is tracked in [docs/staged-roadmap.md](docs/staged-roadmap.md).
