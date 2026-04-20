# Configuration Reference

IsoGraph uses dataclass-based typed configuration models in
`isograph.workflow.config`.

## Command Configs

- `BenchmarkCommandConfig`
  Controls suite generation, backend selection, report locations, real-data freeze
  settings, and backend-specific config blocks.
- `FitCommandConfig`
  Controls baseline fitting for a prepared dataset bundle.
- `CompareCommandConfig`
  Controls report or snapshot comparison output paths.

## Backend Configs

- `BaselineModelConfig`
  Sparse partial-correlation baseline with residualization and trait-association defaults.
- `LatentModelConfig`
  Factor Analysis denoising plus partial-correlation inference.
- `GraphModelConfig`
  Latent model extended with graph-Laplacian smoothing.
- `VaeModelConfig`
  Variational autoencoder backend with early stopping, latent-dimension controls, and
  optional checkpoint output.

## Real-Data and Stability Configs

- `RealDataFreezeConfig`
  Points at the BrainSeq-style count and annotation tables used by `freeze-real`.
- `StabilitySelectionConfig`
  Controls alpha-grid search for real-data edge stability.

## Default Files

The repository ships with these YAML entry points:

- `configs/benchmark.yaml`
- `configs/fit.yaml`
- `configs/compare.yaml`
- `configs/stage3_graph.yaml`
- `configs/stage4_vae.yaml`

Use them as stable defaults and supply Hydra overrides after `--` on the CLI.
