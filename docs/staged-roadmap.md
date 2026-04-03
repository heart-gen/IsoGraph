# IsoGraph Staged Roadmap

## Summary

| Stage | Status | Deliverables | Gate | Evidence | Owner | Last Commit |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | in_progress | Package skeleton, CLI, typed configs, benchmark harness, real-data freeze pipeline, tests, CI, tracking | `core_v1` can be generated and validated; tests pass | pending | kynon | pending |
| 1 | blocked | Deterministic baseline network and benchmark runner | Recovery/stability/runtime gates pass on `core_v1` | pending | kynon | pending |
| 2 | blocked | Probabilistic gene-aware latent model | Beats stage 1 on required scenarios | pending | kynon | pending |
| 3 | blocked | Graph-aware priors/regularization | Beats stage 1 and stays stable | pending | kynon | pending |
| 4 | blocked | VAE extension | Clears benchmark gates beyond stage 2/3 | pending | kynon | pending |

## Update Rule

Only mark a stage `complete` after:

1. The gate listed for that stage passes.
2. The benchmark report path is recorded below.
3. The commit SHA and, if used, MLflow run ID are recorded below.

## Next Unlocked Stage

`Stage 0`: finish the benchmark-first infrastructure and freeze `core_v1`.

## Stage 0

### Deliverables

- Package layout with `io`, `features`, `models`, `evaluation`, and `workflow`.
- Hydra configs and Pydantic validation.
- Benchmark generators for `toy_v1` and `medium_v1`.
- Pure-Python freeze pipeline for `real_caudate_aa_v1`.
- CLI commands: `freeze-real`, `benchmark`, `fit`, `compare`, `export`.
- CI workflow and dedicated conda environment.
- Unit and property-based tests around invariants.

### Commands

```bash
isograph freeze-real
isograph benchmark
pytest
```

### Expected Artifacts

- `benchmarks/datasets/core_v1/toy_v1`
- `benchmarks/datasets/core_v1/medium_v1`
- `benchmarks/datasets/core_v1/real_caudate_aa_v1`
- `artifacts/reports/core_v1-benchmark.json`

### Gate

- Environment builds in a fresh conda env, and CI validates Python `3.11` through `3.14`.
- Validation catches malformed manifests and mismatched matrix/table shapes.
- Synthetic datasets and the real-data subset freeze successfully.
- CI passes on Python `3.11` and `3.12`.

## Stage 1

### Deliverables

- Deterministic baseline feature pipeline.
- Sparse partial-correlation network inference.
- Module detection and module-trait association.
- Benchmark scoring against simulation truth and real-data stability.

### Gate

- `toy_v1` module recovery is exact.
- `medium_v1` clears the predefined recovery thresholds.
- Repeated runs are deterministic.
- Runtime and memory stay within budget.

## Milestone Log

| Date | Stage | Commit | MLflow Run | Benchmark Report | Runtime/Memory | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| pending | 0 | pending | pending | pending | pending | pending |
