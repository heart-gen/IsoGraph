# Limitations and Reproducibility

## Current Product Boundaries

- `benchmark` is centered on the bundled `core_v1` and `scale_v1` fixture suites and the
  real-data freeze workflow.
- `fit` currently exposes only the baseline backend for user-supplied bundles.
- `freeze-real` is tailored to the repository's BrainSeq-style input tables rather than
  arbitrary cohort layouts.
- The VAE and GPU-latent backends require a separate PyTorch installation.
  IsoGraph installs `mpmath` for modern SymPy compatibility, but PyTorch is
  intentionally left to the user because CPU/GPU/CUDA builds are system-specific.
- The WGCNA backend requires R with the `WGCNA` package installed and `Rscript` on `PATH`
  (called via subprocess — no Python R bridge required).

## Data Assumptions

- Models operate on transcript-count matrices aligned to a transcript feature table.
- Trait and covariate analysis is only performed for columns that actually exist in the
  sample table.
- `export` expects a dataset bundle that includes a gene table so the gene count can be
  reported.

## Scale Considerations

- The VAE backend is the recommended choice at high gene counts (6 000–12 000 genes,
  25:1–50:1 genes-to-samples ratios). It has been validated on the `scale_v1` suite with
  recovery ≥ 0.90 on all three scale fixtures.
- The GPU-latent backend uses the Woodbury identity to avoid the p×p covariance matrix
  during FA fitting (O(n²k) vs O(n³)), and is much faster than sklearn FA at scale.
  However, the Woodbury precision matrix is low-rank by construction (rank = k). At high
  p/n ratios BIC selects few components, insufficient to separate all modules. Recovery
  on `xlarge_v1` (6 000 genes) is ~0.04 and on `xxlarge_v1` (12 000 genes) is ~0.42.
  **Use the VAE backend for datasets above 2 000 genes.** Use `alpha_percentile` (e.g.
  99.5) instead of a fixed `alpha` when using GPU-latent at scale.
- The WGCNA backend uses blockwise mode automatically for datasets above 5 000 genes;
  edge tables are not populated in blockwise mode.

## Reproducibility Design

- The fixture suite is benchmark-first and stage-aware.
- Snapshot outputs are named deterministically by stage, fixture, backend, version, and seed.
- The real-data freeze path caches sample selection, projected gene counts, transcript
  count partitions, and frozen fixtures under `benchmarks/cache/real_data/`.
- CI validates the supported Python range and the test suite on every push to `main`.
