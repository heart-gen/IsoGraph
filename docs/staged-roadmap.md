# IsoGraph Staged Roadmap

## Summary

| Stage | Status | Deliverables | Promotion Gate | Required Evidence | Owner | Last Commit |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | complete | Package skeleton, CLI, typed configs, benchmark harness, frozen fixtures, tests, CI, tracking | Fresh install + config validation + smoke workflows + deterministic snapshot pass | CI run, smoke-test log, snapshot diff report, fixture manifest | kynon | ddd5644 |
| 1 | complete | Deterministic baseline feature pipeline, sparse network/module workflow, benchmark runner | Full `core_v1` baseline benchmark clears recovery, stability, and runtime gates | Locked benchmark report, baseline artifacts, seed-stability report | kynon | 98ef297 |
| 2 | complete | Probabilistic gene-aware latent model | Beats or matches Stage 1 on required scenarios and remains calibrated/stable | Comparative benchmark report vs Stage 1, calibration report, ablation report | kynon | 56e4119 |
| 3 | complete | Graph-aware priors/regularization | Improves switch/splicing recovery or interpretability without destabilizing runtime/calibration | Comparative benchmark report vs Stage 2, graph ablation report, prior-edge diagnostics | kynon | cc5f72f |
| 4 | complete | VAE backend | Clears pre-specified gains beyond Stage 2/3 and is reproducible across seeds | Comparative benchmark report, seed-sensitivity report, latent diagnostics, checkpoint manifest | kynon | 6cbaa55 |
| 5 | complete | WGCNA comparison benchmark on simulated data | IsoGraph (best backend) matches or exceeds WGCNA module recovery on `core_v1` synthetic fixtures | Side-by-side recovery table, runtime comparison, manuscript-ready figures | kynon | — |
| 6 | complete | Large-scale fixtures (xlarge_v1 6k, xxlarge_v1 12k, xxlarge_stress_v1 12k); VAE architecture scaling; WGCNA blockwise upgrade | VAE recovers modules ≥ 0.90 on xlarge and xxlarge fixtures; VAE default backend | Benchmark reports, gate tests locked | kynon | — |
| 7 | removed | GPU-accelerated FA backend (Woodbury identity + BIC selection) — removed after benchmarking showed poor module recovery (0.089–0.241) vs. VAE (0.960–0.972) with no runtime advantage | N/A — backend removed | N/A | kynon | — |
| 9 | planned | Multiplex switch/abundance module calibration: channel edge policies, role-aware metrics, tuned configs, and first-class `multiplex_v1` gates | VAE/graph/latent multiplex benchmarks clear role-aware recovery gates without giant-component or edge-density failures | Multiplex benchmark reports, role-specific recall tables, config ablations, WGCNA calibration comparison | kynon | — |

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

## Current Status

Stages 0–6 and 8A–8E are complete. Stage 7 (gpu_latent) was removed. See the Milestone
Log for details. Stage 8 (module explanation) is complete through sub-stage 8E (Captum
Integrated Gradients). Stage 9 is planned to harden mixed switch/abundance module
inference and benchmark reporting.

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

- `artifacts/benchmarks/stage4_vae/medium_v1/` ✓
- `artifacts/benchmarks/stage4_vae/core_v1/` ✓
- `artifacts/reports/stage4-vs-stage3.json` ✓
- `artifacts/reports/stage4-seed-sensitivity.json` ✓
- `artifacts/reports/stage4-latent-diagnostics.json` ✓
- `artifacts/checkpoints/stage4_vae/` (generated on demand via checkpoint_dir config)
- `snapshots/stage4_core_v1_vae_v1_seed0000/` (deferred; not required for promotion)

### Promotion Gate

- The VAE backend shows a locked, pre-specified gain beyond the best earlier backend.
- Gains replicate across multiple seeds.
- Training is stable enough to support reproducible releases.
- Checkpoints reload successfully and produce compatible outputs.
- The VAE remains optional; IsoGraph stays usable without it.

---

---

## Stage 5 — WGCNA Comparison Benchmark

### Objective

Establish IsoGraph's performance relative to WGCNA on the same locked synthetic
fixtures used throughout Stages 1–4. This is the primary manuscript evidence
that IsoGraph recovers biologically relevant co-expression modules as well as or
better than the field-standard method, particularly on nonlinear and noisy data
regimes where WGCNA is expected to struggle.

### Deliverables

- WGCNA runner integrated into the benchmark harness (subprocess calling an R script via `Rscript`).
- Side-by-side module recovery table: WGCNA vs IsoGraph (best backend per fixture) on all `core_v1` fixtures.
- Runtime and memory comparison.
- Manuscript-ready summary table and figures.

### Recommended Checklist

- [ ] WGCNA can be called reproducibly from the benchmark harness (fixed seed, soft-thresholding power selection documented).
- [ ] WGCNA runner produces a `module_table` in the same schema as IsoGraph backends.
- [ ] `module_recovery_score` is applied identically to both methods.
- [ ] Recovery comparison covers all `core_v1` fixtures: toy_v1, medium_v1, realistic_v1, realistic_unequal_v1, noisy_v1, large_v1, nonlinear_v1.
- [ ] WGCNA soft-thresholding power is selected per fixture (scale-free topology criterion, R² ≥ 0.85), not hard-coded.
- [ ] Runtime and memory overhead for WGCNA is measured and reported.
- [ ] IsoGraph VAE backend meets or exceeds WGCNA recovery on nonlinear_v1 and noisy_v1 (the fixtures most likely to differentiate methods).
- [ ] Results are reproducible: same R seed and Python seed produce identical output.
- [ ] Comparison report is archived as `artifacts/reports/stage5-vs-wgcna.json`.
- [ ] Manuscript table drafted from the report.

### Recommended Commands

```bash
# Run WGCNA on core_v1
isograph benchmark --config-name stage5_wgcna

# Compare IsoGraph VAE vs WGCNA
isograph compare \
  --reference artifacts/reports/stage5_wgcna-benchmark.json \
  --candidate artifacts/reports/stage4_vae-benchmark.json \
  --output-path artifacts/reports/stage5-vs-wgcna.json
```

### Expected Artifacts

- `artifacts/benchmarks/stage5_wgcna/core_v1/`
- `artifacts/reports/stage5_wgcna-benchmark.json`
- `artifacts/reports/stage5_wgcna-runtime-memory.json`
- `artifacts/reports/stage5-vs-wgcna.json`
- `configs/stage5_wgcna.yaml`

### Promotion Gate

- WGCNA runner integrates cleanly with the benchmark harness.
- Recovery scores are computed with the same metric as all prior stages.
- IsoGraph (VAE backend) matches or exceeds WGCNA on ≥ 5 of 7 synthetic fixtures.
- IsoGraph shows clear advantage on nonlinear_v1 (VAE recovery 0.958 vs WGCNA expected < 0.5 on radial structure).
- Runtime comparison is documented and acceptable for manuscript claims.

---

## Stage 6 — Large-Scale VAE Architecture Scaling

### Objective

Establish that the VAE backend (with properly scaled architecture) recovers co-expression
modules at 25:1 and 50:1 genes-to-samples ratios, representative of unfiltered bulk RNA-seq
gene counts (12k–24k genes). Promote VAE to the default production backend.

### New Fixtures (`scale_v1` suite)

- **`xlarge_v1`**: 6 000 genes / 240 samples (25:1 ratio), 12 power-law modules, ~15.5% switching, NB dispersion=7.
- **`xxlarge_v1`**: 12 000 genes / 240 samples (50:1 ratio), 16 power-law modules, 25% switching, NB dispersion=7.

Both generated by `generate_scale_suite()` — kept separate from `generate_core_suite()` to preserve test speed.

### Key Changes

- `VaeModelConfig` docstring extended with n_genes-based hidden_dim guidance.
- Auto batch_size: when `batch_size=None` and `n_genes > 2000`, VAE sets `min(64, n_samples // 4)` automatically.
- Default `BenchmarkCommandConfig.backend` changed from `"baseline"` to `"vae"`.
- `prepare_scale_suite()` added to `runner.py`; dispatched when `dataset_suite == "scale_v1"`.

### Recommended Commands

```bash
# Calibration run (sets recovery baseline for gate locking)
isograph benchmark --config-name stage6_vae_xlarge

# After calibration: set _GATE_XLARGE / _GATE_XXLARGE in test_stage6_gates.py,
# then run fast + slow gate tests
pytest tests/test_stage6_gates.py -v
pytest tests/test_stage6_gates.py -v -m slow
```

### Expected Artifacts

- `artifacts/reports/stage6_vae_xlarge-benchmark.json`
- `artifacts/reports/stage6_vae_xlarge-runtime-memory.json`
- `configs/stage6_vae_xlarge.yaml`

### Promotion Gate

- VAE recovery on `xlarge_v1` ≥ calibration-locked threshold (observed − 0.10).
- VAE recovery on `xxlarge_v1` ≥ calibration-locked threshold (observed − 0.10).
- No regression on Stage 4 gates (`test_stage4_gates.py` passes).
- Default backend is `"vae"` with no API breakage.

---

## Stage 7 — GPU Latent Backend (removed)

The GPU linear latent backend (`gpu_latent`) planned for Stage 7 was implemented and
then removed. It achieved 0.089–0.241 module recovery (10–20× worse than VAE), with no
runtime advantage and 64–128 GB GPU memory requirements at scale. VAE remains the
recommended default. The `gpu_latent` backend and its benchmark infrastructure have been
fully excised from the codebase (PR #6).

---

## Stage 8 — Module Explanation

### Objective

Given a fitted IsoGraph artifact and one or more selected modules, identify the
transcript-level and gene-local features that explain each module. The key biological
object is a within-gene isoform switch — a module driver should be a directional
transcript-usage contrast within a gene, not just a transcript with high marginal
correlation.

### Sub-Stages

| Stage | Scope | Status |
|-------|-------|--------|
| 8A | Backend-agnostic tables + CLI + gate tests | **complete** |
| 8B | Publication-ready plots | **complete** |
| 8C | Annotation/consequence table integration | **complete** |
| 8D | VAE decoder attribution (optional, checkpoint required) | **complete** |
| 8E | Captum integrated gradients (optional, `isograph[torch-explain]`) | **complete** |

### Stage 8A: Module Explanation MVP

**New module**: `src/isograph/explain/`

**CLI**:
```bash
isograph explain-module \
  --artifact-dir artifacts/fits/manual \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --module-ids M000 M001 \
  --output-dir artifacts/explain/run1
```

**Python API**:
```python
from isograph.explain import explain_module, ExplainConfig
results = explain_module(
    artifact_dir="artifacts/fits/manual",
    feature_table=feature_df,
    feature_meta=meta_df,
    module_ids=["M000"],
)
```

**Inputs**:
- `artifact_dir`: must contain `modules.parquet` and `feature_scores.parquet`
- `feature_table`: `pd.DataFrame` with sample IDs as index (or `sample_id` column), feature IDs as columns
- `feature_meta`: columns `[feature_id, gene_id, feature_type]` required; `[gene_name, transcript_id, exon_id, event_id, source_coordinate]` optional
- `module_score_table` (optional): override computed eigengenes with precomputed scores

**Outputs per module** in `{output_dir}/{module_id}/`:
- `gene_driver_table.parquet` — `[gene_id, r, pvalue, qvalue, n_samples, missing_fraction]`, sorted by `|r|`
- `transcript_polarity_table.parquet` — `[feature_id, gene_id, [transcript_id], r, pvalue, qvalue, n_samples, missing_fraction, switch_strength]`; `switch_strength` = `max(r) − min(r)` within gene
- `high_vs_low_table.parquet` — `[feature_id, gene_id, mean_high, mean_low, delta, se, tstat, pvalue, n_high, n_low, missing_fraction]`
- `{output_dir}/module_explanation_manifest.json` (always written)

**Validation**:
- Hard fail: missing artifact files, unknown module IDs, missing required feature_meta columns, duplicate sample/feature IDs, <10 overlapping samples
- Soft warn: missing optional columns, dropped samples from alignment

**Gate test**: `tests/test_stage8a_gates.py` (28 tests, all fast except `@pytest.mark.slow` CLI roundtrip)

### Stage 8B: Plots

Per-module publication-ready figures:
- Top driver genes barplot (|r| with 95% CI, colored by polarity)
- Transcript usage gradient plot: x = module score, y = transcript usage, one smoothed trend per transcript per gene, top N driver genes
- Positive vs. negative driver heatmap
- Optional consequence barplot (after 8C)

Controlled by `ExplainConfig(plot=True)`. Plot functions accept DataFrames and output paths; no global state.

### Stage 8C: Annotation and Consequence Integration

Accept an optional annotation/consequence table and merge it into the explain output.
Consequence types to support (flexible column names, normalized at load time):

- coding_change, utr_change, nmd_sensitivity
- domain_gain_loss, signal_peptide_change, coding_noncoding_switch
- alt_first_last_exon, cassette_exon, intron_retention
- alt_5_splice, alt_3_splice

Input-compatible with IsoformSwitchAnalyzeR export format (import/export, not
re-implementation). See: https://bioconductor.org/packages/release/bioc/vignettes/IsoformSwitchAnalyzeR/

### Stage 8D: VAE Decoder Attribution ✓ complete

Only when a VAE checkpoint is available in `artifact_dir`. Perturbs the module-associated
latent dimension (selected by argmax |Pearson r| with module eigengene) and decodes back
into gene space via finite-difference Jacobian to identify high-confidence drivers.

High-confidence filter (three criteria, all required):
- association FDR ≤ threshold (default 0.05) — from `gene_driver_table.qvalue`
- |decoded_delta| ≥ top-percentile threshold (default 90th) among module genes
- sign(decoded_delta × sign(latent_r)) agrees with sign(r) — direction-corrected for
  latent dims that are anti-correlated with the module eigengene

**Implementation:**
- `src/isograph/explain/vae_attribution.py` — `compute_decoder_jacobian()`, `filter_vae_drivers()`
- `src/isograph/explain/config.py` — `ExplainConfig.vae_attribution`, `vae_fdr_threshold`,
  `vae_percentile_threshold`, `vae_perturbation_eps`; `ExplainResult.vae_drivers`
- CLI flags: `--vae-attribution`, `--vae-fdr-threshold`, `--vae-percentile-threshold`
- Per-module output: `{module_id}/vae_drivers.parquet`; manifest key `vae_attribution_available`
- Gate tests: `tests/test_stage8d_gates.py` (34 tests, all pass)

**Accuracy (VAE fit, seed=42, scripts/eval_stage8d_accuracy.py):**

| Fixture | member_auc | switch_vae | switch_r | sign_agree | prec@filter |
|---|---|---|---|---|---|
| toy_v1 (24g) * | 0.594 | — | — | 1.000 | — |
| medium_v1 (400g) * | 0.786 | — | — | 1.000 | — |
| realistic_v1 (200g) | 0.809 | 1.000 | 1.000 | 1.000 | 0.714 |
| noisy_v1 (300g) | 0.977 | — ¹ | — ¹ | 1.000 | 1.000 |
| nonlinear_v1 (200g) | 0.438 | 1.000 | 0.999 | 0.933 | 0.400 |
| large_v1 (800g) | 0.899 | 0.888 | 0.885 | 0.919 | 0.833 |
| xlarge_mini (600g) | 0.941 | 1.000 | 0.758 | 1.000 | 0.700 |
| xxlarge_mini (400g) | 0.984 | — ¹ | — ¹ | 0.875 | 1.000 |

_* toy/medium: all genes switch, switch_auc undefined. ¹ NaN: fitted modules contained
only switching genes (accurate fit). member_auc = AUC(|decoded_delta|, module membership)
over all genes. switch_vae = AUC(|decoded_delta|, truth_switch) within module genes._

Key finding: VAE attribution matches or exceeds Pearson |r| on linear fixtures
(xlarge_mini: switch_vae=1.000 vs switch_r=0.758). Nonlinear (radial/product) modules
are not well-captured by single-dim perturbation (member_auc=0.438).

### Stage 8E: Captum Integrated Gradients

Requires optional install: `pip install isograph[torch-explain]` (adds `captum`).

Use Captum's Integrated Gradients to attribute module eigengene prediction to individual
transcript features via the VAE encoder. Treats the encoder as a black-box mapping
feature_table → latent and integrates gradients from a baseline (zero usage vector) to
the observed input. Returns `DataFrame(feature_id, ig_score)` sorted by |ig_score|.

Unlike decoder Jacobian (8D), IG attributes the full encoder nonlinearity; complementary
to association-based approaches in 8A.

### Recommended Commands

```bash
pytest tests/test_stage8a_gates.py -v
pytest tests/test_stage8a_gates.py -v -m slow
```

### Promotion Gate (8A)

- All 28 gate tests pass.
- No regressions on existing stage gate tests.
- `isograph explain-module --help` shows all flags.

---

## Stage 9 — Multiplex Switch/Abundance Calibration

### Objective

Make mixed switch/abundance module discovery a first-class, calibrated workflow rather
than an incidental consequence of adding abundance channels. Stage 9 focuses on the
remaining gap exposed by `multiplex_v1`: latent/graph models under-include
abundance-only genes unless the graph becomes dense, while WGCNA can over-include
abundance genes and fail its own soft-threshold calibration.

### Deliverables

- Configurable channel edge policies for `switch-switch`, `switch-abundance`,
  `abundance-switch`, and `abundance-abundance` edges.
- Backend-specific defaults for multiplex channel weights and edge inclusion rules.
- Role-aware benchmark metrics:
  - role-specific recall for `switch_only`, `abundance_only`, `coupled`, and `discordant`.
  - module purity by role and truth module.
  - edge-density and giant-component penalties.
  - WGCNA soft-threshold fit warnings in benchmark reports.
- Tuned `multiplex_v1` configs for VAE, graph, and latent backends.
- First-class benchmark configs and gates for `multiplex_v1`.

### Recommended Checklist

- [ ] Add typed config fields for channel edge weights and edge-type inclusion policies.
- [ ] Apply the edge policy consistently across baseline, latent, graph, VAE, and WGCNA projections.
- [ ] Add role-specific recall, abundance recall, switch recall, module purity, and edge-density metrics.
- [ ] Include WGCNA soft-threshold diagnostics in benchmark reports and gate warnings.
- [ ] Add `configs/stage9_multiplex_vae.yaml`.
- [ ] Add `configs/stage9_multiplex_graph.yaml`.
- [ ] Add `configs/stage9_multiplex_latent.yaml`.
- [ ] Tune `n_components`, `alpha`, and channel weights on `toy_multiplex_v1`,
  `medium_multiplex_v1`, `noisy_multiplex_v1`, and `large_multiplex_v1`.
- [ ] Lock role-aware gates once the tuned configs are stable.

### Candidate Promotion Gates

- `medium_multiplex_v1` recovery ≥ 0.70 for VAE and graph.
- `noisy_multiplex_v1` recovery ≥ 0.70 for VAE and graph.
- `large_multiplex_v1` recovery ≥ 0.80 for VAE.
- Abundance recall ≥ 0.70 on medium/noisy/large multiplex fixtures for VAE.
- No fixture has an uncontrolled giant component under the tuned default configs.
- Benchmark reports include role-specific recall and WGCNA soft-threshold diagnostics.

### Recommended Commands

```bash
isograph benchmark --config-name stage9_multiplex_vae
isograph benchmark --config-name stage9_multiplex_graph
isograph benchmark --config-name stage9_multiplex_latent
pytest tests/test_multiplex_features.py tests/test_multiplex_synthetic_fixtures.py -v
```

### Expected Artifacts

- `benchmarks/datasets/multiplex_v1/`
- `artifacts/reports/stage9_multiplex_vae-benchmark.json`
- `artifacts/reports/stage9_multiplex_graph-benchmark.json`
- `artifacts/reports/stage9_multiplex_latent-benchmark.json`
- `artifacts/reports/stage9-multiplex-role-metrics.json`
- `artifacts/reports/stage9-multiplex-config-ablation.json`

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
| 2026-04-19 | 4 | — | N/A | artifacts/reports/stage4-vs-stage3.json, stage4-seed-sensitivity.json, stage4-latent-diagnostics.json | — | Missing Stage 4 evidence artifacts generated; nonlinear_v1 multi-seed mean=0.972±0.010; no posterior collapse on any fixture |
| 2026-04-23 | 5 | — | N/A | artifacts/reports/stage5_wgcna-benchmark.json, stage5-vs-wgcna.json | artifacts/reports/stage5_wgcna-runtime-memory.json | VAE wins 6/7 core_v1 fixtures vs WGCNA (signed network); nonlinear_v1 VAE=0.958 vs WGCNA=0.877; WGCNA edge on large_v1 (0.874 vs 0.795) |
| 2026-04-23 | 6 | — | N/A | artifacts/reports/stage6_vae_xlarge-benchmark.json | artifacts/reports/stage6_vae_xlarge-runtime-memory.json | xlarge_v1 recovery=1.0 (415s), xxlarge_v1 recovery=1.0 (933s); gates locked at 0.90; no collapsed dims; VAE is now default backend |
| 2026-04-25 | 6b | — | N/A | artifacts/reports/stage6_scale_comparison_vae-benchmark.json, stage6_scale_comparison_wgcna-benchmark.json, stage6-scale-vae-vs-wgcna.json | — | xxlarge_stress_v1 added (24 modules, min=30, max=140, ~10% switching, dispersion=15, confounder=0.55); VAE=1.0 vs WGCNA=0.822 on stress fixture; blockwiseModules upgrade for WGCNA at 12k genes |
| 2026-04-25 | 7 | — | N/A | artifacts/reports/stage7_gpu_latent-benchmark.json | artifacts/reports/stage7_gpu_latent-runtime-memory.json | GpuLatentNetworkModel: Woodbury FA + BIC selection; toy_v1=1.0, medium_v1=1.0, realistic_v1=1.0, realistic_unequal_v1=1.0 — matches Stage 2 latent exactly; _GATE_TOY/MED locked at 0.95; xlarge scale gates pending |

| 2026-05-02 | 8A | — | N/A | — | — | Module explanation MVP: explain_module() API, isograph explain-module CLI, gene_driver/transcript_polarity/high_vs_low tables, 28 gate tests + 7 accuracy tests all pass; gene-driver AUC=1.000 and switch_strength AUC=1.000 on realistic_v1 baseline fit |
| 2026-05-02 | 8D | — | N/A | scripts/eval_stage8d_accuracy.py | attr_time ≤ 0.08s per run | VAE decoder attribution: compute_decoder_jacobian + filter_vae_drivers; 34 gate tests pass; member_auc 0.786–0.984 on linear fixtures; switch_vae=1.000 on realistic/xlarge (vs switch_r=0.758 on xlarge); nonlinear member_auc=0.438 (expected: single-dim perturbation cannot capture distributed nonlinear modules); sign fix: direction-corrected by latent_r sign |
| 2026-05-03 | 8E | — | N/A | — | — | Captum Integrated Gradients encoder attribution: compute_integrated_gradients; 30 gate tests pass; IG completeness axiom verified; batched IG exploits sample independence in encoder; shared _select_module_latent_dim helper extracted from 8D; torch-explain optional group added to pyproject.toml |

## Current Recommendation

Stage 7 (gpu_latent) has been removed. See the Stage 7 section above for details.
VAE (Stage 4/6) is the current recommended default backend.
Stage 8 (module explanation) is in progress; Stages 8A-8E are complete.
Stage 9 should be the next development focus before further real-data interpretation:
calibrate multiplex switch/abundance edge policies, role-aware metrics, and
`multiplex_v1` benchmark gates.
