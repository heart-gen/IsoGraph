# Limitations and Reproducibility

## Current Product Boundaries

- `benchmark` is centered on the bundled `core_v1`, `scale_v1`, and `multiplex_v1`
  fixture suites and the real-data freeze workflow.
- `fit` supports all five backends (`baseline`, `latent`, `graph`, `vae`, `wgcna`); VAE
  is the default. The `freeze-real` workflow is tailored to BrainSEQ-style source tables.
- The VAE backend (and optional attribution features) requires a separate PyTorch
  installation. IsoGraph installs `mpmath` for modern SymPy compatibility, but PyTorch is
  intentionally left to the user because CPU/GPU/CUDA builds are system-specific.
- Captum Integrated Gradients (`--integrated-gradients`) requires the optional
  `torch-explain` dependency group: `pip install isograph[torch-explain]`.
- The WGCNA backend requires R with the `WGCNA` package installed and `Rscript` on `PATH`
  (called via subprocess — no Python R bridge required).

## Data Assumptions

- Models operate on transcript-count matrices aligned to a transcript feature table.
- To enable the abundance channel (multiplex mode), the dataset bundle must include a
  `gene_counts` matrix. Without it, IsoGraph runs in switch-only mode.
- When running `explain-module` on multiplex artifacts, `feature_scores.parquet` must
  contain a `feature_type` column (`"switch"` or `"abundance"`). This is populated
  automatically by all backends when multiplex channels are active.
- Trait and covariate analysis is only performed for columns that actually exist in the
  sample table.
- `export` expects a dataset bundle that includes a gene table so the gene count can be
  reported.

## Scale Considerations

- The VAE backend is the recommended choice at high gene counts (6 000–12 000 genes,
  25:1–50:1 genes-to-samples ratios). It has been validated on the `scale_v1` suite with
  recovery ≥ 0.90 on all three scale fixtures.
- The optional `xxlarge_multiplex_v1` fixture has 12,000 genes and 240 samples. It is
  generated only when explicitly requested with `fixture_filter=xxlarge_multiplex_v1`.
- The WGCNA backend uses blockwise mode automatically for datasets above 5 000 genes;
  edge tables are not populated in blockwise mode. At 12k multiplex scale it may require
  a larger `wgcna.timeout_seconds` setting.

## Reproducibility Design

- Snapshot outputs are named deterministically by stage, fixture, backend, version, and seed.
- The real-data freeze path caches sample selection, projected gene counts, transcript
  count partitions, and frozen fixtures under `benchmarks/cache/real_data/`.
- Generated benchmark datasets and bulky per-fixture artifact directories are ignored by
  git; compact JSON reports under `artifacts/reports/` are the intended tracked evidence.
- CI validates the supported Python range and the test suite on every push to `main`.
