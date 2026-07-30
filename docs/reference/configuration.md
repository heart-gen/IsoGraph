# Configuration Reference

IsoGraph uses dataclass-based typed configuration models in
`isograph.workflow.config`.

## Command Configs

- `BenchmarkCommandConfig`
  Controls suite generation, backend selection, report locations, real-data freeze
  settings, and backend-specific config blocks. The default backend is `"vae"`.
- `FitCommandConfig`
  Controls fitting a prepared dataset bundle with any backend (`baseline`, `latent`,
  `graph`, `vae`, `wgcna`); the default backend is `"vae"`.
- `CompareCommandConfig`
  Controls report or snapshot comparison output paths.

## Backend Configs

- `BaselineModelConfig`
  Sparse partial-correlation baseline with residualization and trait-association defaults.
- `LatentModelConfig`
  Factor Analysis denoising plus partial-correlation inference. Supports cross-validated
  or fixed component count selection.
- `GraphModelConfig`
  Latent model extended with graph-Laplacian smoothing.
- `VaeModelConfig`
  Variational autoencoder backend with early stopping, latent-dimension controls, and
  optional checkpoint output. See the `hidden_dim` docstring for gene-count guidance.
- `WgcnaModelConfig`
  WGCNA backend wrapping R's `blockwiseModules`. Configures soft-thresholding power,
  minimum module size, merge cut height, network type, and subprocess timeout.

## Real-Data and Stability Configs

- `RealDataFreezeConfig`
  Points at the BrainSeq-style count and annotation tables used by `freeze-real`.
- `StabilitySelectionConfig`
  Controls alpha-grid search for real-data edge stability.

## Default Config Files

The repository ships with these YAML entry points:

| File | Suite | Backend |
|---|---|---|
| `configs/benchmark.yaml` | `core_v1` | `vae` (default) |
| `configs/fit.yaml` | — | `vae` (default) |
| `configs/compare.yaml` | — | — |
| `configs/stage3_graph.yaml` | `core_v1` | `graph` |
| `configs/stage4_vae.yaml` | `core_v1` | `vae` |
| `configs/stage5_wgcna.yaml` | `core_v1` | `wgcna` |
| `configs/stage6_vae_xlarge.yaml` | `scale_v1` | `vae` |
| `configs/stage6_scale_comparison_vae.yaml` | `scale_v1` | `vae` |
| `configs/stage6_scale_comparison_wgcna.yaml` | `scale_v1` | `wgcna` |
| `configs/stage9_multiplex_vae.yaml` | `multiplex_v1` | `vae` |
| `configs/stage9_multiplex_graph.yaml` | `multiplex_v1` | `graph` |
| `configs/stage9_multiplex_latent.yaml` | `multiplex_v1` | `latent` |
| `configs/stage9_multiplex_wgcna.yaml` | `multiplex_v1` | `wgcna` |
| `configs/stress_multiplex_xxlarge_vae.yaml` | `multiplex_v1` | `vae` |
| `configs/stress_multiplex_xxlarge_wgcna.yaml` | `multiplex_v1` | `wgcna` |
Use them as stable entry points and supply Hydra overrides after `--` on the CLI.

## Multiplex-Specific Fields

VAE, graph, and latent configs can enable multiplex edge policies with:

- `allow_abundance_abundance` — include abundance-abundance edges instead of requiring
  abundance-only genes to connect through switch-active genes.
- `alpha_switch` — threshold for switch-switch feature edges.
- `alpha_switch_grid` — optional grid used to select the switch threshold that avoids
  switch-switch giant components (the switch-channel counterpart of `alpha_abundance_grid`).
- `alpha_abundance` — fixed threshold for abundance-abundance feature edges.
- `alpha_abundance_grid` — optional grid used to select the smallest abundance threshold
  that avoids merging baseline switch modules.

For very large multiplex fixtures, prefer a fixed `alpha_abundance` because grid
selection repeats the O(feature²) graph projection for each candidate threshold.

## VAE Stability and Reliability Controls

`VaeModelConfig` exposes several opt-in fields (all off/neutral by default) for hard
cohorts. They are not set in `configs/fit.yaml`, so they take their dataclass defaults
unless you override them:

- `grad_clip_norm` (default `None`) — clips the global gradient norm before each optimizer
  step to tame early exploding-gradient steps. A divergence guard (non-finite validation
  loss → restore best checkpoint and stop) is always active regardless of this setting.
- `residualize_composition` (default `False`) — regress `residualize_covariates` out of each
  gene's CLR composition *before* the switch PC1 is derived, instead of out of the collapsed
  switch score afterward. Robust to confounds that rotate a gene's switch axis (e.g. 3′
  degradation). See [Residualization is a discovery-only knob](#residualization-is-a-discovery-only-knob-vae-backend).
- `switch_reliability_weighting` (default `False`) — down-weight switch-switch edges by
  per-gene reliability so unreliable genes fall back to the abundance channel. Source is
  chosen with `switch_reliability_source`: `"degradation"` (needs `degradation_covariate`)
  or the covariate-free `"estimability"` (tuned by `switch_estimability_min_minor_usage`).
  `switch_reliability_floor` and `switch_reliability_power` shape the weight curve.
- `grey_min_intra_degree` (default `0`) — WGCNA-style grey-module rejection: iteratively drop
  genes whose intra-module degree is below this `k`, leaving them unassigned.
- `leiden_resolution` (default `5.0`) — Leiden community-detection resolution; higher gives
  more, smaller modules. The data-driven giant-component cap is automatic (the former
  `leiden_max_giant_frac` knob is deprecated).

## Residualization Is a Discovery-Only Knob (VAE Backend)

On the `vae` backend, `residualize_covariates` (and `residualize_composition`) affect
**module discovery only**. The VAE and Leiden clustering embed the residualized matrix, but
the persisted `feature_scores` keep the **raw, pre-residualization** values. Consequences:

- Module assignments and edges reflect the residualized matrix, exactly as before.
- `feature_scores.parquet`, module eigengenes, and therefore `traits.parquet` are computed
  from raw values. The built-in trait test is **not** covariate-adjusted.
- Covariates should enter inference exactly once — in your own downstream model. Previously
  `feature_scores` held the residualized matrix, so a downstream model that re-adjusted the
  same covariates double-residualized them.

Inspect the effect with [`FitArtifacts.residualization_qc`](artifacts.md#residualization_qc).

```{warning}
This split currently applies to the `vae` backend only. The `baseline`, `graph`, `latent`,
and `wgcna` backends still persist the **residualized** matrix in `feature_scores`, so on
those backends a downstream model that re-adjusts the same covariates will double-adjust
them. They also do not emit `residualization_qc`. If you switch backends, check which
convention applies before running a covariate-adjusted downstream model.
```

`build_design_matrix` now emits a `RuntimeWarning` when a name in `residualize_covariates`
is missing from the sample table, or is present but constant (no variance to regress out).
Both cases are still skipped rather than raising, but they are no longer silent.

## Per-Fixture Overrides

All backends support per-fixture config overrides in `BenchmarkCommandConfig`:

- `fixture_model_overrides` — baseline
- `fixture_latent_overrides` — latent
- `fixture_graph_overrides` — graph
- `fixture_vae_overrides` — vae
- `fixture_wgcna_overrides` — wgcna

Values are partial field dicts merged with `dataclasses.replace` before the fit.
