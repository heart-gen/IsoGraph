# Backend Reference

IsoGraph exposes five model backends, each selectable via `backend=<name>` in the
benchmark config or as a Hydra override on the CLI.

## Baseline

`BaselineNetworkModel` is the deterministic regression target. It computes gene-level
switch coordinates, optionally residualizes covariates, estimates a sparse partial
correlation network, and extracts modules as connected components. By default, graph
inference uses switch-channel partial correlations only. Abundance channels are computed
and stored in `feature_scores` (for trait associations and role classification) but do
not drive edge selection unless multiplex mode is enabled (`allow_abundance_abundance=True`
or `alpha_abundance_grid`).

Use it when you want:

- the simplest reproducible backend
- the current custom-data CLI path through `isograph fit`
- a stable reference point for regression testing

## Latent

`LatentNetworkModel` adds sklearn Factor Analysis denoising before partial-correlation
network inference. By default, FA denoising and partial-correlation inference operate on
switch-only channels; abundance channels are in `feature_scores` but are not included in
the network computation unless multiplex mode is enabled. Component count is selected by
cross-validated log-likelihood (default) or fixed. Also records calibration metrics:
held-out log-likelihood, reconstruction RMSE, mean noise variance, and convergence status.

Use it when you want:

- a more noise-tolerant backend than the baseline
- automatic or fixed latent-dimensionality control
- stability selection for real data without ground-truth modules

> **Memory note:** The latent backend constructs a full p×p covariance matrix
> internally. For datasets with >> features (roughly > 1 000 genes), this becomes
> memory-prohibitive. Use the VAE backend for large feature spaces.

## Graph

`GraphNetworkModel` extends the latent backend with graph-Laplacian smoothing over a
gene graph prior to Factor Analysis. Like the latent backend, Laplacian smoothing and FA
operate on switch-only channels by default; set `allow_abundance_abundance=True` or
`alpha_abundance_grid` to include abundance channels in network computation.

Use it when you want:

- graph-aware regularization
- diagnostics on graph structure and smoothing strength
- a bridge between the latent and graph-prior formulations

## VAE (default)

`VaeNetworkModel` is the default production backend and the most flexible option. It uses
a variational autoencoder to learn a nonlinear low-dimensional representation of the
feature matrix, then infers modules from Pearson correlations on the decoded signal.
For multiplex runs, the feature matrix includes abundance and switch channels. Supports
early stopping, posterior-collapse detection, latent-dimension grid search, and optional
checkpoint saving.

Use it when you want:

- the best out-of-the-box module recovery on realistic bulk RNA-seq fixtures
- nonlinear latent representations
- operation at 6 000–12 000 gene scale (25:1–50:1 genes-to-samples)
- multi-channel multiplex discovery: abundance channel (all genes) + switch channel
  (multi-isoform genes); per-channel thresholds (`alpha_switch`, `alpha_abundance`)
  with auto-calibration via `alpha_abundance_grid`
- checkpointed model state (`vae_checkpoint.pt`), required for VAE decoder attribution
  (`--vae-attribution`) and Captum Integrated Gradients (`--integrated-gradients`)

Requires PyTorch. Install a build appropriate for your CPU/GPU/CUDA stack before
use. IsoGraph installs `mpmath` for modern SymPy compatibility, but it does not
install PyTorch automatically.

## WGCNA

`WgcnaNetworkModel` wraps R's `WGCNA::blockwiseModules` for direct benchmark comparison.
For datasets above 5 000 genes the runner uses blockwise mode automatically (avoids full
O(n²) TOM matrix). Edge tables are populated only in non-blockwise mode. The subprocess
timeout is configurable through `WgcnaModelConfig.timeout_seconds`.

Use it when you want:

- a standard community comparison baseline
- signed or unsigned weighted correlation network analysis
- a baseline comparison for multiplex stress runs, where over-segmentation should be
  assessed alongside recovery

Requires R with the `WGCNA` package installed and `Rscript` on `PATH`. The backend
calls R via subprocess — no Python R bridge is needed.

## Important Notes

The `benchmark` and `fit` commands can both drive all five backends. VAE is the default
for both. For `fit` with Hydra overrides, append `--` followed by `<backend>.<field>=<value>`
(e.g. `-- vae.hidden_dim=256`).

Linear backends (baseline, latent, graph) default to switch-only channel graph inference.
Abundance channels are always computed and appear in `feature_scores`, but graph edges are
inferred from switch partial correlations only unless multiplex mode is active. Set
`allow_abundance_abundance=True` or `alpha_abundance_grid=[...]` to enable
abundance-abundance edges; this activates the full multiplex edge-policy (cross-channel
edges between dual-channel genes are always suppressed).

All backends support the multiplex edge-policy fields: `allow_abundance_abundance`,
`alpha_switch`, `alpha_abundance`, `alpha_abundance_grid`. The VAE backend supports
`vae_checkpoint.pt` saving for decoder and encoder attribution. All backends populate
`module_gene_roles.parquet` with channel role assignments.
