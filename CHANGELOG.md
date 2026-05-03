# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] -- 2026-05-03

### Added

- **`isograph explain-module` CLI** (Stage 8A) — explain fitted modules at
  transcript-feature resolution. Produces `gene_driver_table.parquet`,
  `transcript_polarity_table.parquet`, and `high_vs_low_table.parquet` per module, plus a
  shared `module_explanation_manifest.json`. Python API: `isograph.explain.explain_module`.
- **Publication-ready explanation plots** (Stage 8B) — top-driver barplot, transcript
  usage gradient plot, and positive/negative driver heatmap. Enabled with `--plot`;
  format controlled by `--output-format png|pdf`.
- **`isograph annotate-structure` CLI** (Stage 8C) — annotate transcript switch pairs
  with GTF-derived structural labels (first/last exon changes, CDS/UTR shifts, biotype
  switches, shared exon fraction). Supports GENCODE and Ensembl GTF conventions; GTF
  parse cache (`--gtf-cache`) avoids re-parsing on repeated runs. Output integrates with
  `isograph explain-module --annotation-table`.
- **VAE decoder attribution** (Stage 8D) — `--vae-attribution` flag on `explain-module`
  computes a finite-difference Jacobian via the VAE decoder to identify high-confidence
  module drivers. Requires a `vae_checkpoint.pt` in the artifact directory. Writes
  `vae_drivers.parquet` per module.
- **Captum Integrated Gradients attribution** (Stage 8E) — `--integrated-gradients` flag
  attributes module eigengene prediction to transcript features via the VAE encoder using
  Captum IG. Requires `pip install isograph[torch-explain]`. Writes `ig_attributions.parquet`
  per module. Baseline options: `zero` (default) or `mean`.

### Removed

- **`gpu_latent` backend** — removed after benchmarking showed 10–20× worse module
  recovery than the VAE backend (0.089–0.241 vs. 0.960–0.972) with no runtime
  advantage and large GPU memory requirements (64–128 GB). Use `backend=vae` instead.

### Changed

- **`fit` command now supports all backends** — `isograph fit` accepts `--backend
  baseline|latent|graph|vae|wgcna`; VAE is the default (`backend: vae` in `fit.yaml`).
  Previously documented as baseline-only.
- **Latent backend memory warning** — documentation now notes that the CPU latent
  backend forms a p×p covariance matrix and becomes memory-prohibitive for datasets
  with >> features (> ~1 000 genes); VAE is recommended for large feature spaces.

---

## [0.1.2] — 2026-04-29

### Changed

- **Generalized trait associations** — `trait_columns` is now fully data-driven. Any
  column in the sample table can be listed as a trait: numeric columns use Pearson
  correlation and binary categorical columns use a Welch t-test. Previously, `Dx` and
  `Age` were hardcoded and required to be present.
- **Shared `compute_trait_associations` utility** — duplicated `_trait_associations`
  logic has been consolidated into a single function in `models/base.py`. All backends
  (baseline, latent, graph, VAE, GPU-latent, WGCNA) now delegate to this shared
  implementation.
- **Default `trait_columns` updated** — the built-in default changed from
  `["Dx", "Age"]` to `["Age"]` to avoid assuming a diagnosis column is present.

### CI

- Added **Python 3.14** to the test matrix.
- Removed the separate `test-torch` CI job; PyTorch-gated tests are now handled by
  fixture skipping in the main matrix.

---

## [0.1.1] — 2026-04-25

### Added

- **`eigengene_table` in `FitArtifacts`** — all backends now compute and return a
  per-module eigengene matrix (modules × samples) alongside the trait association table.
- **Snapshot export** — `isograph freeze-real` / `isograph benchmark` now writes
  `eigengene_table.parquet` to the output directory when at least one module is detected.
- **CHANGELOG.md** and PyPI installation documentation.

### Fixed

- Corrected documentation for the WGCNA backend: `rpy2` is **not** required; the
  backend calls `Rscript` directly and expects R and the `WGCNA` package in `PATH`.

---

## [0.1.0] — 2026-04-25

Initial release of **IsoGraph**, a Python library for discovering isoform-switch and
splicing co-expression modules from bulk RNA-seq data. Combines compositional
transcript-usage modeling with splice-graph-aware latent network inference to recover
gene-module structure and trait associations.

### Added

**Network backends** — five selectable inference strategies:
- `baseline` — Deterministic sparse network (fast reference)
- `latent` — Factor Analysis + LedoitWolf partial correlation with cross-validation
- `graph` — Graph-Laplacian-smoothed latent model
- `vae` *(default)* — PyTorch nonlinear VAE with early stopping and posterior-collapse diagnostics
- `wgcna` — R `blockwiseModules` wrapper for direct WGCNA comparison

**Compositional transforms** — CLR and logit normalization for transcript-usage proportions

**Benchmarking framework** — fixture-driven recovery scoring across synthetic datasets
(24–12,000 genes) and real BrainSeq-style inputs with snapshot regression testing

**CLI** — `isograph benchmark`, `isograph fit`, `isograph freeze-real`, `isograph compare`,
and `isograph export`; Hydra configuration with `--` override syntax

**Artifact export** — reproducible modules, edges, traits, and calibration reports

### Performance (VAE backend on `core_v1` fixtures)

| Fixture       | Genes | Recovery | Runtime |
|---------------|------:|:--------:|--------:|
| toy_v1        |    24 |   1.00   |   7.5 s |
| medium_v1     |   400 |   1.00   |   4.2 s |
| realistic_v1  |   200 |   1.00   |   1.7 s |
| large_v1      |   800 |   0.80   |   3.8 s |

### Requirements

Python 3.11–3.14. PyTorch and R are optional dependencies required only for the
`vae` and `wgcna` backends respectively.
