# IsoGraph Staged Roadmap

## Summary

| Stage | Status | Deliverables | Promotion Gate | Required Evidence | Owner | Last Commit |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | complete | Package skeleton, CLI, typed configs, benchmark harness, frozen fixtures, tests, CI, tracking | Fresh install + config validation + smoke workflows + deterministic snapshot pass | CI run, smoke-test log, snapshot diff report, fixture manifest | kynon | ddd5644 |
| 1 | complete | Deterministic baseline feature pipeline, sparse network/module workflow, benchmark runner | Full `core_v1` baseline benchmark clears recovery, stability, and runtime gates | Locked benchmark report, baseline artifacts, seed-stability report | kynon | 98ef297 |
| 2 | complete | Probabilistic gene-aware latent model | Beats or matches Stage 1 on required scenarios and remains calibrated/stable | Comparative benchmark report vs Stage 1, calibration report, ablation report | kynon | 56e4119 |
| 3 | complete | Graph-aware priors/regularization | Improves switch/splicing recovery or interpretability without destabilizing runtime/calibration | Comparative benchmark report vs Stage 2, graph ablation report, prior-edge diagnostics | kynon | cc5f72f |
| 4 | complete | VAE backend | Clears pre-specified gains beyond Stage 2/3 and is reproducible across seeds | Comparative benchmark report, seed-sensitivity report, latent diagnostics, checkpoint manifest | kynon | 6cbaa55 |

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

All stages (0–4) are complete. No further stages are currently planned.

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
- Pure-Python freeze pipeline for `realmini_v1`, with PSI as the only splicing feature type and a repo-local real-data cache.
- Artifact manifest and tracking utilities.
- Conda environment and CI workflow.
- Unit, property, regression, and smoke tests.

### Recommended Checklist

- [x] Fresh environment install succeeds.
- [x] CLI imports and `--help` works.
- [x] `fit.yaml`, `benchmark.yaml`, and `compare.yaml` all validate.
- [x] Invalid configs fail with clear messages.
- [x] Synthetic fixtures cover invariants and edge cases.
- [x] End-to-end CLI smoke tests pass on `tiny_v1`, `medium_v1`, and `real_caudate_aa_v1`.
- [x] Expected artifacts are written and validated for all fixture types (gene, transcript, PSI assays).
- [x] Deterministic snapshot comparison passes against `stage0_toy_v1_baseline_v1_seed0000` and `stage0_real_caudate_aa_v1_baseline_v1_seed0000` (committed reference snapshots).
- [x] A frozen `real_caudate_aa_v1` fixture (PSI-only splicing) is built from local BrainSeq data and validates with gene, transcript, and PSI assays. *(note: roadmap used placeholder name `realmini_v1`; canonical name is `real_caudate_aa_v1`)*
- [x] `pytest` passes locally.
- [x] GitHub Actions passes on supported Python versions.
- [x] `environment.yml` creates a fresh working conda environment from scratch.
- [x] `README.md` includes the quickstart install + smoke-test commands.

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
- `realmini_v1` freeze succeeds and validates with PSI-only splicing artifacts.
- End-to-end CLI smoke tests pass on `tiny_v1` and `realmini_v1`.
- Deterministic compare against `stage0_realmini_baseline_v1_seed0000` passes.
- CI passes on all supported Python versions (`3.11`, `3.12`, `3.13`, and `3.14`).

### Stage 0 Notes

- Synthetic fixture coverage plus a frozen real-data mini fixture is sufficient before Stage 1.
- A full `core_v1` benchmark run is not required to complete Stage 0.
- `realmini_v1` intentionally excludes junctions; PSI is the canonical Stage-0 splicing representation.
- The real-data freeze path should use a repo-local cache for filtered samples, projected gene counts, partitioned transcript counts, and frozen mini fixtures so later Stage-0 smoke runs do not rescan remote sources.

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

- [x] Feature construction is deterministic under a fixed seed.
- [x] Sample-order permutation invariance is tested.
- [x] Genes with one retained isoform are handled explicitly.
- [x] Zero/near-zero and all-missing edge cases fail cleanly.
- [x] `toy_v1` truth-recovery tests pass exactly (recovery == 1.0, alpha=0.05).
- [x] `medium_v1` recovery metrics clear the predefined threshold table (≥ 0.875, alpha=0.02; achieved 0.875).
- [x] Baseline backend completes the full `core_v1` run.
- [x] Repeated `core_v1` runs with the same seed produce identical summaries.
- [x] Runtime and memory stay within the Stage 1 budget (tracked via tracemalloc; peak ≤ 9MB per fixture).
- [x] Baseline report and artifacts are archived under `artifacts/benchmarks/stage1_baseline/` and `snapshots/stage1_core_v1_baseline_v1_seed0007/`.
- [x] The compare command shows no regression relative to the Stage 1 baseline snapshot.

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

- [x] Latent backend runs on `toy_v1`, `medium_v1`, and full `core_v1` suite.
- [x] Convergence/failure states are logged clearly (ConvergenceWarning captured; `n_iter`, `converged` in calibration dict; per-fixture `run.log`).
- [x] Calibration metrics are implemented and reported (`mean_log_likelihood`, `reconstruction_rmse`, `mean_noise_variance`, `n_components_used`, `n_components_selected_by`).
- [x] Posterior summaries are stable across repeated runs (determinism gate test passes).
- [x] Ablation runs exist for key modeling choices (`artifacts/reports/stage2_latent-ablation.json`; grid density, CV folds, fixed-k vs auto-k).
- [x] Latent backend matches or exceeds Stage 1 on all must-pass scenarios (toy_v1=1.0, medium_v1=1.0 vs Stage 1 medium_v1=0.875).
- [x] No regression on baseline scenarios.
- [x] Runtime and memory overhead documented (`artifacts/reports/stage2_latent-runtime-memory.json`).
- [x] Comparative benchmark report vs Stage 1 archived (`artifacts/reports/stage2-vs-stage1.json`).
- [x] Backend can be disabled cleanly via `backend=baseline` config without changing the public API.
- [x] Stability selection (`evaluation/selection.py`) is implemented, documented, and gate-tested.
- [x] Stability selection recommends a correct alpha on synthetic data with known modules (verified in `test_stage2_gates.py` and `test_realistic_fixtures.py`).
- [x] Benchmark runner writes per-fixture stability JSON when `run_stability_selection=True`.
- [x] `realistic_v1` fixture added to `core_v1` suite: 50 % non-switching genes, variable isoforms (2–5), NB overdispersion, shared confounder, equal module sizes.
- [x] `realistic_unequal_v1` fixture added: same as `realistic_v1` but with power-law module sizes [38,25,17,12,8] to isolate module-size imbalance.
- [x] Latent recovery gates cleared without n_components leakage: CV selection auto-identifies correct k; realistic_v1=1.0, realistic_unequal_v1=1.0.
- [x] Baseline failure on realistic fixtures is documented as expected behaviour in `tests/test_realistic_fixtures.py`.
- [x] README includes Stage 2 worked example with realistic_v1 (stability selection workflow, calibration interpretation).

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

- [x] Graph construction is deterministic and versioned.
- [x] Edge types are validated and documented.
- [x] Empty/sparse/dense graph edge cases are handled cleanly.
- [x] Graph prior ablations are implemented.
- [x] Gains are localized to biologically motivated scenarios, not only to overfitting on one fixture.
- [x] Same public API as earlier backends is preserved.
- [x] Prior-edge diagnostics are exportable.
- [x] Graph backend matches or exceeds the best earlier backend on must-pass scenarios.
- [x] Runtime blow-up is quantified and acceptable.

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

- [x] VAE backend trains successfully on `tiny_v1`, `medium_v1`, and `core_v1`.
- [x] Checkpoint save/load is verified.
- [x] Seed sensitivity is quantified across multiple runs.
- [x] Posterior collapse or unstable training is explicitly monitored.
- [x] Latent diagnostics are exported.
- [x] VAE beats the best earlier backend on pre-specified targets.
- [x] Improvement is not limited to one random seed.
- [x] Runtime and memory costs are acceptable for the observed gain.
- [x] Public API remains stable across backends.
- [x] Non-VAE backend remains available as a first-class option.

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
| 2026-04-18 | 0 | ddd5644b1e9e5f60d3d40e4001bdedfc3047a20c | N/A | snapshots/stage0_toy_v1_baseline_v1_seed0000, snapshots/stage0_real_caudate_aa_v1_baseline_v1_seed0000 | N/A | Package skeleton, CLI, configs, benchmark harness, frozen fixtures, CI, tracking infra complete |
| 2026-04-18 | 1 | 98ef29792cff8db5a557148e3ee545d66c72078d | N/A | artifacts/reports/stage1_baseline-benchmark.json | artifacts/reports/stage1_baseline-runtime-memory.json | toy_v1=1.0, medium_v1=0.875, real=complete; peak mem ≤ 9MB |
| 2026-04-18 | 2 | 56e41191fa963d31dc3ec88dc37fb9f08acd4201 | N/A | artifacts/reports/stage2_latent-benchmark.json | artifacts/reports/stage2_latent-runtime-memory.json | toy_v1=1.0, medium_v1=1.0 (↑0.125 vs Stage 1), realistic_v1=1.0, realistic_unequal_v1=1.0; CV component selection; no n_components leakage; delta_recovery vs Stage 1: medium_v1 +0.125 |
| 2026-04-18 | 3 | cc5f72f0680286806852586e20069946f3b11371 | N/A | artifacts/reports/stage3_graph-benchmark.json | artifacts/reports/stage3_graph-runtime-memory.json | toy_v1=1.0, medium_v1=1.0; graph-Laplacian priors; Stage 3 complete |
| 2026-04-19 | 4 | 6cbaa55232fbbbc7e431d05e02c6eaaeb8dca3d9 | N/A | artifacts/reports/stage4_vae-benchmark.json | artifacts/reports/stage4_vae-runtime-memory.json | toy_v1=1.0, medium_v1=1.0, nonlinear_v1=0.958 (+0.423 vs Stage 3); unified Pearson inference; no inference_mode param |
| 2026-04-19 | 4 | 99c90b9 | N/A | artifacts/reports/stage4_vae-benchmark.json | artifacts/reports/stage4_vae-runtime-memory.json | latent_dim_grid RMSE-threshold sweep; hidden_dim default 64→128; user guidance docstring; all gate_failures=[] |

## Current Recommendation

Finish 0 first, but keep the artifact and snapshot names aligned with the later
stages now so that Stage 1 through Stage 4 can reuse the same benchmark and
comparison machinery without restructuring the repo.
