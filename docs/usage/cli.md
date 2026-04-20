# CLI Reference

IsoGraph exposes a single CLI entry point:

```bash
isograph --help
```

The current subcommands are `benchmark`, `freeze-real`, `fit`, `compare`, and `export`.

## Overrides

`benchmark` and `freeze-real` use Hydra-style overrides after `--`:

```bash
isograph benchmark -- backend=latent fixture_filter=medium_v1 stage_name=stage2_docs
```

## `benchmark`

Run the bundled fixture suite, or a filtered subset, through a selected backend.

Example:

```bash
isograph benchmark -- \
  backend=graph \
  fixture_filter=realistic_v1 \
  stage_name=graph_docs
```

Behavior:

- Generates the synthetic `core_v1` fixtures as needed.
- Freezes the real fixture unless `fixture_filter` selects only synthetic datasets.
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

Fit the deterministic baseline backend on a prepared dataset bundle.

Example:

```bash
isograph fit \
  --dataset-path benchmarks/datasets/core_v1/toy_v1 \
  --output-dir artifacts/fits/toy_v1
```

Outputs:

- `modules.parquet`
- `edges.parquet`
- `traits.parquet`
- `feature_scores.parquet`
- `fit_config.json`

`fit` currently uses `BaselineNetworkModel`; custom-data runs for the latent, graph, and
VAE backends require the Python API.

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
  --reference artifacts/reports/stage1_baseline-benchmark.json \
  --candidate artifacts/reports/stage2_latent-benchmark.json
```

## `export`

Write a JSON summary of a prepared dataset bundle.

Example:

```bash
isograph export \
  --dataset-path benchmarks/datasets/core_v1/toy_v1 \
  --output-path artifacts/reports/toy_v1-summary.json
```
