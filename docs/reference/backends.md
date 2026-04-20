# Backend Reference

IsoGraph currently exposes four model backends.

## Baseline

`BaselineNetworkModel` is the deterministic regression target. It computes gene-level
switch coordinates, optionally residualizes covariates, estimates a sparse partial
correlation network, and extracts modules as connected components.

Use it when you want:

- the simplest reproducible backend
- the current custom-data CLI path through `isograph fit`
- a stable reference point for regression testing

## Latent

`LatentNetworkModel` adds Factor Analysis denoising before partial-correlation network
inference. It also records calibration metrics such as held-out log likelihood,
reconstruction RMSE, and mean noise variance.

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

## VAE

`VaeNetworkModel` is the most flexible backend and introduces a neural latent-variable
model with early stopping, posterior-collapse diagnostics, and optional checkpointing.

Use it when you want:

- a nonlinear latent representation
- checkpointed model state
- stage-4 behavior from the benchmark suite

Install PyTorch separately before using the VAE backend.

## Important Current Boundary

The `benchmark` command can drive multiple backends on the bundled suite. The `fit`
command currently runs only the baseline backend on custom bundles.
