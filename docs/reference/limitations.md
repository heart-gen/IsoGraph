# Limitations and Reproducibility

## Current Product Boundaries

- `benchmark` is centered on the bundled fixture suite and real-data freeze workflow.
- `fit` currently exposes only the baseline backend for user-supplied bundles.
- `freeze-real` is tailored to the repository's BrainSeq-style input tables rather than
  arbitrary cohort layouts.
- The VAE backend requires a separate PyTorch installation.

## Data Assumptions

- Models operate on transcript-count matrices aligned to a transcript feature table.
- Trait and covariate analysis is only performed for columns that actually exist in the
  sample table.
- `export` expects a dataset bundle that includes a gene table so the gene count can be
  reported.

## Reproducibility Design

- The fixture suite is benchmark-first and stage-aware.
- Snapshot outputs are named deterministically by stage, fixture, backend, version, and seed.
- The real-data freeze path caches sample selection, projected gene counts, transcript
  count partitions, and frozen fixtures under `benchmarks/cache/real_data/`.
- CI validates the supported Python range and the test suite on every push to `main`.
