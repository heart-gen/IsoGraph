# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-25

Initial release of **IsoGraph**, a Python library for discovering isoform-switch and
splicing co-expression modules from bulk RNA-seq data. Combines compositional
transcript-usage modeling with splice-graph-aware latent network inference to recover
gene-module structure and trait associations.

### Added

**Network backends** — six selectable inference strategies:
- `baseline` — Deterministic sparse network (fast reference)
- `latent` — Factor Analysis + LedoitWolf partial correlation with cross-validation
- `graph` — Graph-Laplacian-smoothed latent model
- `vae` *(default)* — PyTorch nonlinear VAE with early stopping and posterior-collapse diagnostics
- `wgcna` — R `blockwiseModules` wrapper for direct WGCNA comparison
- `gpu_latent` — Woodbury-identity Factor Analysis for memory-efficient large-scale inference

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
