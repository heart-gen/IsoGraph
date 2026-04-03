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

### Roadmap

- [x] Fresh environment install succeeds
- [x] CLI imports and `--help` works
- [ ] `fit.yaml`, `benchmark.yaml`, and `compare.yaml` all validate
- [ ] Invalid configs fail with clear messages
- [ ] Synthetic fixture coverage exercises invariants and edge cases
- [ ] One frozen real-data mini fixture catches schema, metadata, sample-order, and artifact problems
- [ ] End-to-end CLI smoke tests pass on both the synthetic fixture and the frozen real-data mini fixture
- [ ] Expected artifacts are written and validated for both fixture types
- [ ] Deterministic snapshot comparison passes against `snapshots/stage0_realmini_baseline_v1_seed0000`
- [x] `pytest` passes locally
- [ ] GitHub Actions passes on supported Python versions (`3.11` to `3.14`)

### Commands

```bash
isograph freeze-real
isograph benchmark
pytest
```

### Stage 0 Fixtures

- Synthetic fixtures are sufficient for invariant and edge-case coverage before Stage 1; a full `core_v1` benchmark run is not required to unlock Stage 1.
- A frozen real-data mini fixture must exist specifically to catch schema, metadata, sample-order, and artifact-regression failures before larger benchmark runs.

### Deterministic Snapshot

- Named target: `stage0_realmini_baseline_v1_seed0000`
- Reference directory: `snapshots/stage0_realmini_baseline_v1_seed0000`
- Candidate directory: `artifacts/current_run`
- Comparison command: `isograph compare`
- Snapshot names must encode only the load-bearing fields:
  `stage`, `fixture` (`tiny`, `realmini`, `core_v1`), `backend` (`baseline`, `latent`, `graph`, `vae`), `version`, and `seed`
- Required snapshot contents:
  - `manifest.json`
  - `metrics.json`
  - `module_summary.tsv`
  - `switch_features.parquet`
  - `run_config.yaml`

### Expected Artifacts

- `benchmarks/datasets/core_v1/toy_v1`
- `benchmarks/datasets/core_v1/medium_v1`
- `benchmarks/datasets/core_v1/real_caudate_aa_v1`
- `artifacts/reports/core_v1-benchmark.json`

### Gate

- Environment builds in a fresh conda env, and CI validates Python `3.11` through `3.14`.
- Validation catches malformed manifests and mismatched matrix/table shapes.
- Synthetic fixtures cover invariants and edge cases successfully.
- The frozen real-data mini fixture is generated and validated successfully.
- End-to-end CLI smoke tests pass on the synthetic and real-data mini fixtures.
- Deterministic comparison against `snapshots/stage0_realmini_baseline_v1_seed0000` passes.
- CI passes on Python `3.11`, `3.12`, `3.13`, and `3.14`.

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
