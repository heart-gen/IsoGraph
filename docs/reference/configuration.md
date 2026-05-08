# Configuration Reference

IsoGraph uses dataclass-based typed configuration models in
`isograph.workflow.config`.

## Command Configs

- `BenchmarkCommandConfig`
  Controls suite generation, backend selection, report locations, real-data freeze
  settings, and backend-specific config blocks. The default backend is `"vae"`.
- `FitCommandConfig`
  Controls baseline fitting for a prepared dataset bundle.
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
| `configs/fit.yaml` | — | baseline |
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
- `alpha_abundance` — fixed threshold for abundance-abundance feature edges.
- `alpha_abundance_grid` — optional grid used to select the smallest abundance threshold
  that avoids merging baseline switch modules.

For very large multiplex fixtures, prefer a fixed `alpha_abundance` because grid
selection repeats the O(feature²) graph projection for each candidate threshold.

## Per-Fixture Overrides

All backends support per-fixture config overrides in `BenchmarkCommandConfig`:

- `fixture_model_overrides` — baseline
- `fixture_latent_overrides` — latent
- `fixture_graph_overrides` — graph
- `fixture_vae_overrides` — vae
- `fixture_wgcna_overrides` — wgcna

Values are partial field dicts merged with `dataclasses.replace` before the fit.
