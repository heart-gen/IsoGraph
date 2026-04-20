# Outputs and Artifacts

IsoGraph writes most outputs into `artifacts/`, `benchmarks/`, or `snapshots/`.

## Dataset Bundles

A prepared dataset bundle contains:

- `manifest.json`
- a sample metadata parquet file
- one or more feature-table parquet files
- one or more dense matrix `.npz` files
- optional truth tables for synthetic fixtures

## Benchmark Outputs

`benchmark` writes:

- per-fixture artifact directories under `artifacts/benchmarks/<stage_name>/<dataset>/`
- benchmark summary JSON under `artifacts/reports/<stage_name>-benchmark.json`
- runtime and memory JSON under `artifacts/reports/<stage_name>-runtime-memory.json`
- calibration JSON when supported by the selected backend
- stability-selection JSON for real-data fixtures when enabled

Each benchmark artifact directory also receives a `run.log` and a snapshot-like set of
tables written through `save_snapshot`.

## Fit Outputs

`fit` writes:

- `modules.parquet`
- `edges.parquet`
- `traits.parquet`
- `feature_scores.parquet`
- `fit_config.json`

## Compare Outputs

`compare` writes a JSON report describing either:

- differences between two snapshot directories, or
- delta metrics between two benchmark summary files

## Export Outputs

`export` writes a JSON summary of a prepared dataset bundle, including sample count, gene
count, and available assays.
