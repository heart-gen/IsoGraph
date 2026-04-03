# IsoGraph Staged Roadmap

## Summary

| Stage | Status | Deliverables | Promotion Gate | Required Evidence | Owner | Last Commit |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | in_progress | Package skeleton, CLI, typed configs, benchmark harness, frozen fixtures, tests, CI, tracking | Fresh install + config validation + smoke workflows + deterministic snapshot pass | CI run, smoke-test log, snapshot diff report, fixture manifest | kynon | pending |
| 1 | blocked | Deterministic baseline feature pipeline, sparse network/module workflow, benchmark runner | Full `core_v1` baseline benchmark clears recovery, stability, and runtime gates | Locked benchmark report, baseline artifacts, seed-stability report | kynon | pending |
| 2 | blocked | Probabilistic gene-aware latent model | Beats or matches Stage 1 on required scenarios and remains calibrated/stable | Comparative benchmark report vs Stage 1, calibration report, ablation report | kynon | pending |
| 3 | blocked | Graph-aware priors/regularization | Improves switch/splicing recovery or interpretability without destabilizing runtime/calibration | Comparative benchmark report vs Stage 2, graph ablation report, prior-edge diagnostics | kynon | pending |
| 4 | blocked | VAE backend | Clears pre-specified gains beyond Stage 2/3 and is reproducible across seeds | Comparative benchmark report, seed-sensitivity report, latent diagnostics, checkpoint manifest | kynon | pending |

## Promotion Rule

Only mark a stage `complete` after:

1. The promotion gate listed for that stage passes.
2. The benchmark report or validation report path is recorded in the milestone log.
3. The commit SHA and, if used, MLflow run ID are recorded in the milestone log.
4. The stage-specific evidence artifacts are committed or archived in the expected location.
5. Any newly introduced config fields, CLI flags, or artifact schemas are documented in `README.md` and `docs/`.

## General Development Policy

- Benchmark-first: every new backend must run through the same locked fixture suite.
- One complexity layer at a time: do not add new biology/modeling ideas and software infrastructure changes in the same milestone.
- Do not retire the previous backend when a new one lands; keep it as a regression target.
- A later stage may only become the default backend after it clears the earlier stage's benchmark suite.
- Snapshot names must encode only the load-bearing fields:
  `stage`, `fixture` (`tiny`, `realmini`, `core_v1`), `backend` (`baseline`, `latent`, `graph`, `vae`), `version`, and `seed`.

## Named Deterministic Snapshots

- Example target: `stage0_realmini_baseline_v1_seed0000`
- Reference directory: `snapshots/<snapshot_name>`
- Candidate directory: `artifacts/current_run`
- Required snapshot contents:
  - `manifest.json`
  - `metrics.json`
  - `module_summary.tsv`
  - `switch_features.parquet`
  - `run_config.yaml`
  - `logs/run.log`

### Snapshot Compare Command

```bash
isograph compare \
  --reference snapshots/stage0_realmini_baseline_v1_seed0000 \
  --candidate artifacts/current_run
```

## Next Unlocked Stage

`Stage 0`: finish infrastructure, frozen fixtures, smoke tests, and deterministic snapshot coverage.

---

## Stage 0 — Infrastructure, Fixtures, and Reproducibility

### Objective

Prove that IsoGraph can be installed from scratch, configured safely, run
end-to-end on tiny fixtures, produce stable artifacts, and fail cleanly on
malformed inputs.

### Deliverables

- Package layout with `io`, `features`, `models`, `evaluation`, and `workflow`.
- CLI commands: `freeze-real`, `benchmark`, `fit`, `compare`, `export`.
- Hydra configs and Pydantic validation.
- Synthetic fixture generators for `tiny_v1` and `medium_v1`.
- Pure-Python freeze pipeline for `realmini_v1`.
- Artifact manifest and tracking utilities.
- Conda environment and CI workflow.
- Unit, property, regression, and smoke tests.

### Recommended Checklist

- [x] Fresh environment install succeeds.
- [x] CLI imports and `--help` works.
- [ ] `fit.yaml`, `benchmark.yaml`, and `compare.yaml` all validate.
- [ ] Invalid configs fail with clear messages.
- [ ] Synthetic fixtures cover invariants and edge cases.
- [ ] A frozen `realmini_v1` fixture catches schema, metadata, sample-order, and artifact issues.
- [ ] End-to-end CLI smoke tests pass on both `tiny_v1` and `realmini_v1`.
- [ ] Expected artifacts are written and validated for both fixture types.
- [ ] Deterministic snapshot comparison passes against `stage0_realmini_baseline_v1_seed0000`.
- [x] `pytest` passes locally.
- [x] GitHub Actions passes on supported Python versions.
- [ ] `environment.yml` creates a fresh working conda environment from scratch.
- [ ] `README.md` includes the quickstart install + smoke-test commands.

### Recommended Commands

```bash
conda env create -f environment.yml
conda activate isograph
pip install -e .
python -m isograph.workflow.cli --help
python -m isograph.workflow.cli fit config=configs/fit.yaml
python -m isograph.workflow.cli benchmark config=configs/benchmark.yaml
python -m isograph.workflow.cli compare config=configs/compare.yaml
pytest -q
```

### Expected Artifacts

- `benchmarks/datasets/tiny_v1/`
- `benchmarks/datasets/medium_v1/`
- `benchmarks/datasets/realmini_v1/`
- `artifacts/smoke/tiny_v1/`
- `artifacts/smoke/realmini_v1/`
- `snapshots/stage0_realmini_baseline_v1_seed0000/`
- `artifacts/reports/stage0-smoke.json`

### Promotion Gate

- Fresh conda environment builds successfully.
- Config validation catches malformed manifests and mismatched matrix/table shapes.
- Synthetic fixtures cover invariants and edge cases successfully.
- `realmini_v1` freeze succeeds and validates.
- End-to-end CLI smoke tests pass on `tiny_v1` and `realmini_v1`.
- Deterministic compare against `stage0_realmini_baseline_v1_seed0000` passes.
- CI passes on all supported Python versions (`3.11`, `3.12`, `3.13`, and `3.14`).

### Stage 0 Notes

- Synthetic fixture coverage plus a frozen real-data mini fixture is sufficient before Stage 1.
- A full `core_v1` benchmark run is not required to complete Stage 0.

---

## Stage 1 — Deterministic Baseline Network Backend

### Objective

Establish the first biologically meaningful backend: a deterministic,
interpretable baseline for switch-aware feature construction, sparse network
inference, module discovery, and association testing.

### Deliverables

- Deterministic feature pipeline for abundance, usage/splicing balances, and switch summaries.
- Baseline model backend under `models/baseline.py`.
- Sparse partial-correlation or equivalent deterministic network inference.
- Module detection and module-trait association.
- Benchmark runner for `core_v1` using the baseline backend.
- Real-data export tables and module summaries.

### Recommended Checklist

- [ ] Feature construction is deterministic under a fixed seed.
- [ ] Sample-order permutation invariance is tested.
- [ ] Genes with one retained isoform are handled explicitly.
- [ ] Zero/near-zero and all-missing edge cases fail cleanly.
- [ ] `toy_v1` truth-recovery tests pass exactly.
- [ ] `medium_v1` recovery metrics clear the predefined threshold table.
- [ ] Baseline backend completes the full `core_v1` run.
- [ ] Repeated `core_v1` runs with the same seed produce identical summaries.
- [ ] Runtime and memory stay within the Stage 1 budget.
- [ ] Baseline report and artifacts are archived under a locked versioned path.
- [ ] The compare command shows no regression relative to the Stage 1 baseline snapshot.

### Recommended Commands

```bash
python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=baseline \
  fixture=toy_v1

python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=baseline \
  fixture=medium_v1

python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=baseline \
  fixture=core_v1 \
  seed=0
```

### Expected Artifacts

- `artifacts/benchmarks/stage1_baseline/toy_v1/`
- `artifacts/benchmarks/stage1_baseline/medium_v1/`
- `artifacts/benchmarks/stage1_baseline/core_v1/`
- `artifacts/reports/stage1-baseline-benchmark.json`
- `snapshots/stage1_core_v1_baseline_v1_seed0000/`
- `artifacts/reports/stage1-runtime-memory.json`

### Promotion Gate

- `toy_v1` module recovery is exact.
- `medium_v1` clears the locked recovery thresholds.
- The full `core_v1` benchmark completes successfully.
- Repeated runs are deterministic for all locked summaries.
- Runtime and memory stay within the Stage 1 budget.
- Stage 1 baseline artifacts are frozen as the primary regression target for later stages.

---

## Stage 2 — Probabilistic Gene-Aware Latent Model

### Objective

Add a probabilistic latent backend that better captures within-gene switching
and uncertainty while preserving calibration, interpretability, and operational stability.

### Deliverables

- `latent` backend with feature-type-aware likelihoods.
- Gene-aware latent representations for abundance and switch/splicing features.
- Calibrated uncertainty summaries.
- Comparative benchmark runner against Stage 1.
- Ablation support for likelihood and shrinkage choices.

### Recommended Checklist

- [ ] Latent backend runs on `tiny_v1`, `medium_v1`, and `core_v1`.
- [ ] Convergence/failure states are logged clearly.
- [ ] Calibration metrics are implemented and reported.
- [ ] Posterior summaries are stable across repeated runs or restarts.
- [ ] Ablation runs exist for key modeling choices.
- [ ] Latent backend matches or exceeds Stage 1 on all must-pass scenarios.
- [ ] Any gains on switching scenarios do not come with unacceptable losses on baseline scenarios.
- [ ] Runtime and memory overhead are documented.
- [ ] Comparative benchmark report vs Stage 1 is archived.
- [ ] The backend can be disabled cleanly via config without changing the public API.

### Recommended Commands

```bash
python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=latent \
  fixture=medium_v1

python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=latent \
  fixture=core_v1 \
  seed=0

python -m isograph.workflow.cli compare \
  --reference artifacts/benchmarks/stage1_baseline/core_v1 \
  --candidate artifacts/benchmarks/stage2_latent/core_v1
```

### Expected Artifacts

- `artifacts/benchmarks/stage2_latent/medium_v1/`
- `artifacts/benchmarks/stage2_latent/core_v1/`
- `artifacts/reports/stage2-vs-stage1.json`
- `artifacts/reports/stage2-calibration.json`
- `artifacts/reports/stage2-ablation.json`
- `snapshots/stage2_core_v1_latent_v1_seed0000/`

### Promotion Gate

- Stage 2 matches or beats Stage 1 on every must-pass benchmark scenario.
- Stage 2 shows a pre-specified improvement on switching/splicing-heavy scenarios.
- Calibration metrics are acceptable and documented.
- Runs are operationally stable and restartable.
- Runtime and memory costs are acceptable relative to the observed gains.

---

## Stage 3 — Graph-Aware Priors and Regularization

### Objective

Inject graph structure explicitly through splice-graph, same-gene, or
biologically informed edges, while preserving the gains from Stage 2 and
improving recovery or interpretability.

### Deliverables

- `graph` backend or graph-regularized mode.
- Support for typed edges: same-gene, shared exon/junction, splice-graph adjacency, optional pathway/RBP priors.
- Prior-edge diagnostics and graph ablation workflows.
- Comparative benchmarks vs Stage 2 and Stage 1.

### Recommended Checklist

- [ ] Graph construction is deterministic and versioned.
- [ ] Edge types are validated and documented.
- [ ] Empty/sparse/dense graph edge cases are handled cleanly.
- [ ] Graph prior ablations are implemented.
- [ ] Gains are localized to biologically motivated scenarios, not only to overfitting on one fixture.
- [ ] Same public API as earlier backends is preserved.
- [ ] Prior-edge diagnostics are exportable.
- [ ] Graph backend matches or exceeds the best earlier backend on must-pass scenarios.
- [ ] Runtime blow-up is quantified and acceptable.

### Recommended Commands

```bash
python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=graph \
  fixture=medium_v1

python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=graph \
  fixture=core_v1 \
  seed=0

python -m isograph.workflow.cli compare \
  --reference artifacts/benchmarks/stage2_latent/core_v1 \
  --candidate artifacts/benchmarks/stage3_graph/core_v1
```

### Expected Artifacts

- `artifacts/benchmarks/stage3_graph/medium_v1/`
- `artifacts/benchmarks/stage3_graph/core_v1/`
- `artifacts/reports/stage3-vs-stage2.json`
- `artifacts/reports/stage3-graph-ablation.json`
- `artifacts/reports/stage3-prior-diagnostics.json`
- `snapshots/stage3_core_v1_graph_v1_seed0000/`

### Promotion Gate

- Graph-aware priors improve recovery or interpretability on pre-specified targets.
- No regression on must-pass baseline scenarios.
- Graph ablations support the claim that the priors are helping.
- Artifact schemas and CLI behavior remain stable.
- Runtime and memory remain acceptable.

---

## Stage 4 — VAE Extension

### Objective

Add a VAE backend only if it yields measurable gains beyond the best non-VAE
backend in benchmark performance, representation quality, or downstream utility.

### Deliverables

- `vae` backend with checkpointed training workflow.
- Latent diagnostics and seed-sensitivity analysis.
- Comparative benchmarks vs the best earlier backend.
- Training/restart/checkpoint manifests.
- Clear fallback path to the deterministic or probabilistic non-VAE backends.

### Recommended Checklist

- [ ] VAE backend trains successfully on `tiny_v1`, `medium_v1`, and `core_v1`.
- [ ] Checkpoint save/load is verified.
- [ ] Seed sensitivity is quantified across multiple runs.
- [ ] Posterior collapse or unstable training is explicitly monitored.
- [ ] Latent diagnostics are exported.
- [ ] VAE beats the best earlier backend on pre-specified targets.
- [ ] Improvement is not limited to one random seed.
- [ ] Runtime and memory costs are acceptable for the observed gain.
- [ ] Public API remains stable across backends.
- [ ] Non-VAE backend remains available as a first-class option.

### Recommended Commands

```bash
python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=vae \
  fixture=medium_v1 \
  seed=0

python -m isograph.workflow.cli benchmark \
  config=configs/benchmark.yaml \
  backend=vae \
  fixture=core_v1 \
  seed=0

python -m isograph.workflow.cli compare \
  --reference artifacts/benchmarks/stage3_graph/core_v1 \
  --candidate artifacts/benchmarks/stage4_vae/core_v1
```

### Expected Artifacts

- `artifacts/benchmarks/stage4_vae/medium_v1/`
- `artifacts/benchmarks/stage4_vae/core_v1/`
- `artifacts/reports/stage4-vs-stage3.json`
- `artifacts/reports/stage4-seed-sensitivity.json`
- `artifacts/reports/stage4-latent-diagnostics.json`
- `artifacts/checkpoints/stage4_vae/`
- `snapshots/stage4_core_v1_vae_v1_seed0000/`

### Promotion Gate

- The VAE backend shows a locked, pre-specified gain beyond the best earlier backend.
- Gains replicate across multiple seeds.
- Training is stable enough to support reproducible releases.
- Checkpoints reload successfully and produce compatible outputs.
- The VAE remains optional; IsoGraph stays usable without it.

---

## Milestone Log

| Date | Stage | Commit | MLflow Run | Benchmark Report | Runtime/Memory | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| pending | 0 | pending | pending | pending | pending | pending |
| pending | 1 | pending | pending | pending | pending | pending |
| pending | 2 | pending | pending | pending | pending | pending |
| pending | 3 | pending | pending | pending | pending | pending |
| pending | 4 | pending | pending | pending | pending | pending |

## Current Recommendation

Finish 0 first, but keep the artifact and snapshot names aligned with the later
stages now so that Stage 1 through Stage 4 can reuse the same benchmark and
comparison machinery without restructuring the repo.
