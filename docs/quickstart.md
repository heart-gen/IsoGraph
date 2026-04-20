# Quickstart

This quickstart validates a fresh checkout on the smallest bundled fixture.

## 1. Activate the Environment

```bash
conda activate isograph
```

## 2. Run a Toy Benchmark

```bash
isograph benchmark -- \
  backend=baseline \
  fixture_filter=toy_v1 \
  stage_name=quickstart_baseline
```

Expected outputs:

- `artifacts/benchmarks/quickstart_baseline/toy_v1/`
- `artifacts/reports/quickstart_baseline-benchmark.json`
- `artifacts/reports/quickstart_baseline-runtime-memory.json`

## 3. Inspect the Result

The benchmark report is a JSON summary with per-dataset metrics such as:

- number of discovered modules
- number of inferred edges
- recovery score when fixture truth is available
- runtime in seconds

## 4. Next Steps

- For command-line reference, continue to [](usage/cli.md).
- For bringing your own dataset into IsoGraph, continue to [](usage/own-data.md).
- For tutorial-style walkthroughs, use the GitHub Wiki.
