# IsoGraph

IsoGraph is a benchmark-first toolkit for gene-aware compositional latent network modeling.
The repository is intentionally organized so benchmark infrastructure and a deterministic
reference baseline exist before later graph-aware or variational models are added.

## Design Rules

- Every model must run on the permanent `core_v1` benchmark suite:
  `toy_v1`, `medium_v1`, `realistic_v1`, `realistic_unequal_v1`, and `real_caudate_aa_v1`.
- Runtime code is split into five layers: `io`, `features`, `models`,
  `evaluation`, and `workflow`.
- The stage-1 baseline stays in the repository as a regression target even if
  later models outperform it.
- Biological sophistication and software-complexity growth should not land in
  the same commit.

## Project Layout

```text
src/isograph/
  io/          dataset manifests, artifacts, real-data freeze pipeline
  features/    residualization and gene-aware feature transforms
  models/      deterministic baseline and future model interfaces
  evaluation/  benchmark runners, metrics, and alpha selection
  workflow/    CLI entry points and typed configs
configs/       Hydra configs
docs/          staged implementation tracker
tests/         unit and property-based tests
```

## Environment

Create a dedicated conda environment for this repository:

```bash
conda env create -f environment.yml
conda activate isograph
```

The environment file installs the project in editable mode with dev dependencies.

## Quickstart

Install and validate a fresh checkout:

```bash
conda env create -f environment.yml
conda activate isograph
isograph --help
pytest -q tests/test_synthetic.py
```

The package is intended to work on Python `3.11` through `3.14`. The conda
environment file pins `3.11` as the canonical local development runtime, while
CI should validate the wider supported range.

If `conda` is not initialized in the current shell, use
`eval "$(conda shell.bash hook)"` first or initialize it according to your
local installation.

## CLI

The package exposes a single CLI with benchmark-first entry points:

```bash
isograph freeze-real
isograph benchmark
isograph fit
isograph compare
isograph export
```

Each command accepts Hydra-style overrides after `--`, for example:

```bash
isograph benchmark -- dataset_suite=core_v1 model.alpha=0.12
```

`freeze-real` uses a repo-local cache by default at
`benchmarks/cache/real_data`. The first population pass may need to read the
upstream BrainSeq sources, but later Stage 0 smoke runs should reuse the local
cached sample selection, projected gene counts, partitioned transcript-count
cache, and frozen `realmini` fixture.

## Alpha Selection for Real Data

When running on real datasets without known module ground truth, the partial-correlation
threshold `alpha` must be chosen empirically. IsoGraph provides **stability selection**
(`isograph.evaluation.selection`) to guide this choice.

### How it works

Stability selection (Meinshausen & Bühlmann, 2010) estimates how reproducibly each
gene–gene edge appears across random subsamples of the data:

1. For each candidate `alpha` in a user-supplied grid, repeat `n_iterations` times:
   - Draw a random subsample of `subsample_fraction` of the samples (without replacement).
   - Fit the model on the subsampled data.
   - Record which gene pairs appear as edges.
2. Compute a **stability score** for each gene pair: the fraction of rounds in which that
   pair appeared as an edge.
3. Count the number of **stable edges** (score ≥ `stability_threshold`, default 0.6) at
   each alpha.
4. Report the coarsest (largest) alpha with at least one stable edge as the
   `recommended_alpha`.

On synthetic data with known modules, within-module edges achieve near-perfect stability
(score ≈ 1.0) while between-module edges have stability ≈ 0.0 at the correct alpha.
This property is verified in the Stage 2 gate tests.

### Using stability selection in Python

```python
from isograph.evaluation.selection import stability_selection
from isograph.models.latent import LatentNetworkModel
from isograph.workflow.config import LatentModelConfig

model = LatentNetworkModel(LatentModelConfig(n_components=7, alpha=0.05))

result = stability_selection(
    model=model,
    transcript_counts=transcript_counts,   # (n_transcripts, n_samples)
    transcript_table=transcript_table,
    sample_table=sample_table,
    alpha_grid=[0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    n_iterations=50,          # higher → more stable estimates
    subsample_fraction=0.8,   # fraction of samples per round
    stability_threshold=0.6,  # minimum fraction to call an edge stable
    seed=0,
)

print(result.recommended_alpha)          # e.g. 0.05
print(result.summary_table())            # DataFrame: alpha, stable_edge_count
```

### Enabling stability selection in the benchmark runner

Set `run_stability_selection=True` (and optionally tune `stability.*`) in
`configs/benchmark.yaml` or via CLI overrides. The runner will run stability selection
on any fixture that lacks ground-truth module labels (i.e., real-data fixtures) and
write a per-fixture JSON report:

```bash
isograph benchmark -- backend=latent stage_name=stage2_latent \
  run_stability_selection=true \
  stability.alpha_grid=[0.005,0.01,0.02,0.05,0.10] \
  stability.n_iterations=50 \
  fixture_filter=real_caudate_aa_v1
```

Output: `artifacts/reports/stage2_latent-real_caudate_aa_v1-stability.json`

```json
{
  "dataset": "real_caudate_aa_v1",
  "backend": "latent",
  "recommended_alpha": 0.05,
  "summary": [
    {"alpha": 0.005, "stable_edge_count": 8100},
    {"alpha": 0.01,  "stable_edge_count": 3200},
    {"alpha": 0.05,  "stable_edge_count": 420},
    {"alpha": 0.10,  "stable_edge_count": 0}
  ]
}
```

The `recommended_alpha` is appended to the benchmark report row for that fixture so
downstream comparisons can reference the alpha that was actually used.

### Interpreting the stability curve

- A **monotonically decreasing** stable-edge count with increasing alpha is the expected
  pattern. The curve reflects sparsity: higher alpha → fewer but more reproducible edges.
- The **recommended alpha** is the coarsest threshold that still produces at least one
  stable edge. A good operating point is typically where the curve shows an elbow — a
  rapid drop in stable-edge count that signals the threshold has exceeded the true signal.
- For real RNA-seq data, edge counts in the hundreds (not thousands) with stability ≥ 0.6
  are a reasonable starting point before biological interpretation.

## Synthetic Benchmark Fixtures

The `core_v1` suite contains four synthetic fixtures with increasing biological realism,
plus the frozen real-data fixture `real_caudate_aa_v1`.

| Fixture | Genes | Samples | Modules | Isoforms/gene | Non-switching | NB counts | Confounder | Module sizes |
|---|---|---|---|---|---|---|---|---|
| `toy_v1` | 24 | 48 | 2 | 2 | 0 % | No | No | Equal |
| `medium_v1` | 400 | 240 | 8 | 2 | 0 % | No | No | Equal |
| `realistic_v1` | 200 | 160 | 5 | 2–5 | 50 % | Yes | Yes | **Equal** |
| `realistic_unequal_v1` | 200 | 160 | 5 | 2–5 | 50 % | Yes | Yes | **[38,25,17,12,8]** |

The two realistic fixtures form a **controlled pair**: `realistic_unequal_v1` differs from
`realistic_v1` only in module-size distribution. This isolates the effect of module-size
imbalance from all other realistic features.

**Known limitation of the Stage 1 baseline on realistic fixtures:** The Ledoit-Wolf
partial-correlation estimator fails to separate modules when 50 % of genes are
non-switching noise. This is expected and documented in `tests/test_realistic_fixtures.py`.
The Stage 2 latent model (FA denoising + partial correlation, `n_components=5, alpha=0.02`)
recovers all equal-sized modules (recovery ≥ 0.875 gate) and meets a stricter gate on
unequal modules (recovery ≥ 0.5).

## Stage 2 Worked Example: realistic_v1

This walkthrough demonstrates the full Stage 2 pipeline using `realistic_v1` as a
proxy for real data (the same workflow applies to `real_caudate_aa_v1`).

### Step 1 — Run the latent benchmark on realistic fixtures

```bash
# Generate datasets and run the latent backend on both realistic fixtures
isograph benchmark -- \
  backend=latent \
  stage_name=stage2_latent \
  fixture_filter=realistic_v1 \
  seed=7
```

Output artifacts:
```
artifacts/benchmarks/stage2_latent/realistic_v1/
artifacts/reports/stage2_latent-benchmark.json
artifacts/reports/stage2_latent-calibration.json   # FA log-likelihood, RMSE, noise variance
```

### Step 2 — Use stability selection to choose alpha (as you would for real data)

`realistic_v1` has ground truth, but we can run stability selection as though it were
unknown to demonstrate the workflow:

```python
from isograph.io.artifacts import load_dataset_bundle
from isograph.models.latent import LatentNetworkModel
from isograph.workflow.config import LatentModelConfig
from isograph.evaluation.selection import stability_selection

bundle = load_dataset_bundle("benchmarks/datasets/core_v1/realistic_v1")
model  = LatentNetworkModel(LatentModelConfig(n_components=5, alpha=0.02))

result = stability_selection(
    model=model,
    transcript_counts=bundle.matrices["transcript_counts"],
    transcript_table=bundle.feature_tables["transcript"],
    sample_table=bundle.sample_table,
    alpha_grid=[0.01, 0.02, 0.05, 0.10],
    n_iterations=50,
    subsample_fraction=0.8,
    stability_threshold=0.6,
    seed=0,
)

print(result.recommended_alpha)  # expected: 0.02
print(result.summary_table())
#    alpha  stable_edge_count
# 0   0.01               2450
# 1   0.02                490
# 2   0.05                  0
# 3   0.10                  0
```

The stability curve drops sharply from alpha=0.01 to alpha=0.02 — the elbow marks the
boundary of reproducible signal. The recommended alpha (last non-zero stable-edge count)
is 0.02.

### Step 3 — Compare latent vs baseline on realistic fixtures

```bash
isograph compare \
  --reference artifacts/benchmarks/stage1_baseline/realistic_v1 \
  --candidate artifacts/benchmarks/stage2_latent/realistic_v1 \
  --output-path artifacts/reports/stage2-vs-stage1-realistic.json
```

The comparison table will show `delta_recovery > 0` — the latent backend substantially
outperforms the baseline on realistic fixtures, demonstrating the practical value of the
probabilistic denoising step.

### Step 4 — Understand the calibration output

The calibration JSON written by the latent benchmark run contains per-fixture diagnostics:

```json
{
  "calibration_by_fixture": [{
    "fixture": "realistic_v1",
    "mean_log_likelihood": -2.31,
    "reconstruction_rmse": 0.42,
    "mean_noise_variance": 0.18,
    "n_components_used": 5
  }]
}
```

| Field | Interpretation |
|---|---|
| `mean_log_likelihood` | Higher (less negative) → FA model fits the data better |
| `reconstruction_rmse` | Lower → denoising is more effective; very low suggests overfit |
| `mean_noise_variance` | Mean per-gene idiosyncratic noise; higher → more gene-specific unexplained variance |
| `n_components_used` | Actual k used (may be < requested if constrained by n_genes or n_samples) |

## Alpha Selection for Real Data

When running on real datasets without known module ground truth, the partial-correlation
threshold `alpha` must be chosen empirically. IsoGraph provides **stability selection**
(`isograph.evaluation.selection`) to guide this choice.

### How it works

Stability selection (Meinshausen & Bühlmann, 2010) estimates how reproducibly each
gene–gene edge appears across random subsamples of the data:

1. For each candidate `alpha` in a user-supplied grid, repeat `n_iterations` times:
   - Draw a random subsample of `subsample_fraction` of the samples (without replacement).
   - Fit the model on the subsampled data.
   - Record which gene pairs appear as edges.
2. Compute a **stability score** for each gene pair: the fraction of rounds in which that
   pair appeared as an edge.
3. Count the number of **stable edges** (score ≥ `stability_threshold`, default 0.6) at
   each alpha.
4. Report the coarsest (largest) alpha with at least one stable edge as the
   `recommended_alpha`.

On synthetic data with known modules, within-module edges achieve near-perfect stability
(score ≈ 1.0) while between-module edges have stability ≈ 0.0 at the correct alpha.
This property is verified in the Stage 2 gate tests.

### Using stability selection in Python

```python
from isograph.evaluation.selection import stability_selection
from isograph.models.latent import LatentNetworkModel
from isograph.workflow.config import LatentModelConfig

model = LatentNetworkModel(LatentModelConfig(n_components=5, alpha=0.02))

result = stability_selection(
    model=model,
    transcript_counts=transcript_counts,   # (n_transcripts, n_samples)
    transcript_table=transcript_table,
    sample_table=sample_table,
    alpha_grid=[0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
    n_iterations=50,
    subsample_fraction=0.8,
    stability_threshold=0.6,
    seed=0,
)

print(result.recommended_alpha)
print(result.summary_table())
```

### Enabling stability selection in the benchmark runner

Set `run_stability_selection=True` in `configs/benchmark.yaml` or via CLI overrides.
The runner will run stability selection on any fixture that lacks ground-truth module
labels (i.e., real-data fixtures) and write a per-fixture JSON report:

```bash
isograph benchmark -- backend=latent stage_name=stage2_latent \
  run_stability_selection=true \
  stability.n_iterations=50 \
  fixture_filter=real_caudate_aa_v1
```

Output: `artifacts/reports/stage2_latent-real_caudate_aa_v1-stability.json`

### Interpreting the stability curve

- A **monotonically decreasing** stable-edge count with increasing alpha is expected.
- The **recommended alpha** is the coarsest threshold with at least one stable edge.
- A good operating point is typically the elbow — where the count drops sharply.
- For real RNA-seq data, stable-edge counts in the hundreds (not thousands) with
  stability ≥ 0.6 are a reasonable starting point before biological interpretation.

## Status

Implementation progress is tracked in [docs/staged-roadmap.md](docs/staged-roadmap.md).
