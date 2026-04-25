# Backend Reference

IsoGraph exposes six model backends, each selectable via `backend=<name>` in the
benchmark config or as a Hydra override on the CLI.

## Baseline

`BaselineNetworkModel` is the deterministic regression target. It computes gene-level
switch coordinates, optionally residualizes covariates, estimates a sparse partial
correlation network, and extracts modules as connected components.

Use it when you want:

- the simplest reproducible backend
- the current custom-data CLI path through `isograph fit`
- a stable reference point for regression testing

## Latent

`LatentNetworkModel` adds sklearn Factor Analysis denoising before partial-correlation
network inference. Component count is selected by cross-validated log-likelihood (default)
or fixed. Also records calibration metrics: held-out log-likelihood, reconstruction RMSE,
mean noise variance, and convergence status.

Use it when you want:

- a more noise-tolerant backend than the baseline
- automatic or fixed latent-dimensionality control
- stability selection for real data without ground-truth modules

## Graph

`GraphNetworkModel` extends the latent backend with graph-Laplacian smoothing over a
gene graph prior to Factor Analysis.

Use it when you want:

- graph-aware regularization
- diagnostics on graph structure and smoothing strength
- a bridge between the latent and graph-prior formulations

## VAE (default)

`VaeNetworkModel` is the default production backend and the most flexible option. It uses
a variational autoencoder to learn a nonlinear low-dimensional representation of the
switch-coordinate matrix, then infers modules from Pearson correlations on the decoded
signal. Supports early stopping, posterior-collapse detection, latent-dimension grid
search, and optional checkpoint saving.

Use it when you want:

- the best out-of-the-box module recovery on realistic bulk RNA-seq fixtures
- nonlinear latent representations
- operation at 6 000–12 000 gene scale (25:1–50:1 genes-to-samples)
- checkpointed model state

Requires PyTorch. Install separately before use.

## WGCNA

`WgcnaNetworkModel` wraps R's `WGCNA::blockwiseModules` for direct benchmark comparison.
For datasets above 5 000 genes the runner uses blockwise mode automatically (avoids full
O(n²) TOM matrix). Edge tables are populated only in non-blockwise mode.

Use it when you want:

- a standard community comparison baseline
- signed or unsigned weighted correlation network analysis

Requires R with the `WGCNA` package installed and `Rscript` on `PATH`. The backend
calls R via subprocess — no Python R bridge is needed.

## GPU Latent

`GpuLatentNetworkModel` is a PyTorch reimplementation of Factor Analysis using the
**Woodbury matrix identity** — the p×p covariance matrix Σ = WW^T + Ψ is never
materialised; all operations are O(n·p·k + k³) where k ≪ p. Component count is
selected by BIC elbow search instead of 5-fold CV, cutting selection time from
O(|grid| × 5 × max_iter) to O(|grid| × max_iter). Per-gene noise variances Ψ
(diagonal of the noise covariance) are available in `calibration` as
`gpu_latent_per_gene_noise_var`.

**Alpha threshold**: a fixed `alpha` or a data-adaptive `alpha_percentile` (e.g. 95.0)
can be used. With `alpha_percentile` the threshold is computed as the Nth percentile of
absolute partial-correlation values, which automatically scales with dataset size and
p/n ratio.

Use it when you want:

- Factor Analysis denoising with explicit per-gene noise-variance diagnostics
- faster component selection than the CV-based latent backend
- memory-efficient FA at high gene counts (avoids dense p×p matrices)

Matches `LatentNetworkModel` recovery on all `core_v1` fixtures. Requires PyTorch.

## Important Current Boundary

The `benchmark` command can drive all six backends on the bundled suites. The `fit`
command currently runs only the baseline backend on custom bundles.
