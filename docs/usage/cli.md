# CLI Reference

IsoGraph exposes a single CLI entry point:

```bash
isograph --help
```

The current subcommands are `benchmark`, `freeze-real`, `fit`, `compare`, and `export`.

## Overrides

`benchmark`, `freeze-real`, and `fit` accept Hydra-style overrides after `--`:

```bash
isograph benchmark -- backend=latent fixture_filter=medium_v1 stage_name=stage2_docs
isograph fit --dataset-path my_cohort --backend vae -- vae.alpha=0.6 vae.hidden_dim=256
```

## `benchmark`

Run the bundled fixture suite, or a filtered subset, through a selected backend.

The default backend is `vae`. Available backends: `baseline`, `latent`, `graph`, `vae`,
`wgcna`, `gpu_latent`.

Examples:

```bash
# VAE on a single fixture (default backend)
isograph benchmark -- fixture_filter=toy_v1 stage_name=vae_toy

# GPU-latent on the full core suite
isograph benchmark -- backend=gpu_latent stage_name=gpu_latent_core

# WGCNA on the scale suite
isograph benchmark --config-name stage6_scale_comparison_wgcna

# Named config for a complete stage
isograph benchmark --config-name stage7_gpu_latent
```

Behavior:

- For `dataset_suite: core_v1` — generates the synthetic `core_v1` fixtures as needed and
  freezes the real fixture unless `fixture_filter` targets only synthetic datasets.
- For `dataset_suite: scale_v1` — generates `xlarge_v1` (6 000 genes), `xxlarge_v1`
  (12 000 genes), and `xxlarge_stress_v1` (12 000 genes, stressed parameters).
- Writes per-fixture artifacts under `artifacts/benchmarks/<stage_name>/`.
- Writes benchmark and runtime summaries under `artifacts/reports/`.
- Writes calibration reports when the selected backend emits calibration metadata.

## `freeze-real`

Freeze the bundled real-data fixture from local source tables.

Example:

```bash
isograph freeze-real --suite-name core_v1
```

The command reads `BenchmarkCommandConfig.real_data` through the benchmark config and
caches intermediate selections under `benchmarks/cache/real_data/`.

## `fit`

Fit any backend on a prepared dataset bundle.

```bash
# VAE (default)
isograph fit \
  --dataset-path benchmarks/datasets/custom/my_cohort_v1 \
  --output-dir artifacts/fits/vae_default

# Baseline
isograph fit \
  --dataset-path benchmarks/datasets/core_v1/toy_v1 \
  --backend baseline \
  --output-dir artifacts/fits/toy_v1

# VAE with Hydra overrides
isograph fit \
  --dataset-path benchmarks/datasets/custom/my_cohort_v1 \
  --backend vae \
  --output-dir artifacts/fits/vae_tuned \
  -- vae.alpha=0.6 vae.hidden_dim=256 vae.n_epochs=400

# GPU-Latent with percentile threshold for large gene counts
isograph fit \
  --dataset-path benchmarks/datasets/custom/my_cohort_v1 \
  --backend gpu_latent \
  -- gpu_latent.alpha_percentile=95.0
```

Available backends: `baseline`, `latent`, `graph`, `vae`, `wgcna`, `gpu_latent`.

Outputs:

- `modules.parquet`
- `edges.parquet`
- `traits.parquet`
- `feature_scores.parquet`
- `calibration.json` (when the backend emits calibration metadata — VAE, latent, GPU-latent)
- `fit_config.json`

Default config values for all backends live in `configs/fit.yaml` and can be
overridden with Hydra syntax after `--`.

## `compare`

Compare either two snapshot directories or two benchmark JSON reports.

Examples:

```bash
isograph compare \
  --reference snapshots/stage0_toy_v1_baseline_v1_seed0000 \
  --candidate artifacts/benchmarks/quickstart_baseline/toy_v1
```

```bash
isograph compare \
  --reference artifacts/reports/stage2_latent-benchmark.json \
  --candidate artifacts/reports/stage7_gpu_latent-benchmark.json
```

## `export`

Write a JSON summary of a prepared dataset bundle.

Example:

```bash
isograph export \
  --dataset-path benchmarks/datasets/core_v1/toy_v1 \
  --output-path artifacts/reports/toy_v1-summary.json
```
