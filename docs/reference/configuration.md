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
  minimum module size, merge cut height, and network type.

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
| `configs/stage6_vae_xlarge.yaml` | `core_v1` | `vae` |
| `configs/stage6_scale_comparison_vae.yaml` | `scale_v1` | `vae` |
| `configs/stage6_scale_comparison_wgcna.yaml` | `scale_v1` | `wgcna` |
Use them as stable entry points and supply Hydra overrides after `--` on the CLI.

## Per-Fixture Overrides

All backends support per-fixture config overrides in `BenchmarkCommandConfig`:

- `fixture_model_overrides` — baseline
- `fixture_latent_overrides` — latent
- `fixture_graph_overrides` — graph
- `fixture_vae_overrides` — vae
- `fixture_wgcna_overrides` — wgcna

Values are partial field dicts merged with `dataclasses.replace` before the fit.
