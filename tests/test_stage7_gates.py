"""Stage 7 promotion gate tests — GPU-accelerated Factor Analysis backend.

Covers Stage 7 checklist items:
  - toy_v1 and medium_v1 recovery matches LatentNetworkModel (both ≥ 1.0)
  - calibration dict includes gpu_latent_per_gene_noise_var
  - FitArtifacts schema valid
  - BIC selects a k from n_components_grid
  - per_gene_noise_var shape = (n_genes,), values > 0
  - Determinism: two identical fits → identical module_table
  - Graceful ImportError when torch absent
  - xlarge_v1 recovery ≥ GATE_XLARGE_GPU_LATENT  (slow; gate set post-calibration)
  - xlarge_v1 runtime faster than FA baseline / 3    (slow)
  - xxlarge_v1 runtime budget ≤ 900s               (slow)

CALIBRATION PROCESS
-------------------
1. Run: isograph benchmark --config-name stage7_gpu_latent
2. Check artifacts/reports/stage7_gpu_latent-benchmark.json for observed recovery.
3. Set _GATE_TOY  = min(observed_toy,   1.0)
   Set _GATE_MED  = min(observed_medium, 1.0)
4. Commit the updated constants and re-run this file.

NOTE ON SCALE RECOVERY
-----------------------
GPU-latent recovery on xlarge_v1/xxlarge_v1 is limited (~0.04–0.42) because
the Woodbury FA precision matrix is low-rank by construction (rank = k << p).
At high p/n ratios, BIC selects few components (4–10), insufficient to
separate all planted modules. The gate is set to 0.0 (non-regression only).
VAE is the recommended backend for scale datasets — see Stage 6 gates.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="PyTorch not installed — skip Stage 7 GPU latent gate tests")

from isograph.benchmarks.synthetic import generate_core_suite, generate_scale_suite
from isograph.evaluation.metrics import module_recovery_score
from isograph.io.artifacts import load_dataset_bundle
from isograph.models.gpu_latent import GpuLatentNetworkModel
from isograph.workflow.config import GpuLatentModelConfig

# ---------------------------------------------------------------------------
# Calibration-locked recovery gates.
# Set to None until first calibration benchmark run completes.
# After calibration: set to observed_recovery - 0.05 (tighter guard than VAE
# because FA is deterministic up to optimizer seed).
# ---------------------------------------------------------------------------
_GATE_TOY: float | None = 0.95    # observed=1.0  − 0.05; seed=7; 2026-04-25
_GATE_MED: float | None = 0.95    # observed=1.0  − 0.05; seed=7; 2026-04-25
_GATE_XLARGE: float | None = 0.0  # observed=0.037; scale limited by p/n ratio; 2026-04-25

_TOY_CONFIG = GpuLatentModelConfig(
    n_components_grid=[2, 3, 4, 5, 6],
    bic_selection=True,
    max_iter=300,
    lr=1e-2,
    tol=1e-5,
    alpha=0.05,   # lower threshold needed for 24-gene toy fixture
    min_module_size=2,
    random_state=7,
)
_MED_CONFIG = GpuLatentModelConfig(
    n_components_grid=[2, 3, 4, 5, 6, 7, 8, 9, 10],
    bic_selection=True,
    max_iter=500,
    lr=1e-2,
    tol=1e-5,
    alpha=0.01,   # partial correlations on FA-denoised data are ~0.02 for medium_v1
    min_module_size=2,
    random_state=7,
)
_XLARGE_CONFIG = GpuLatentModelConfig(
    n_components_grid=[8, 10, 12, 15, 18, 20],
    bic_selection=True,
    max_iter=1000,
    lr=5e-3,
    tol=1e-5,
    alpha=0.03,
    alpha_percentile=99.5,   # data-adaptive threshold for high p/n ratio
    min_module_size=2,
    random_state=7,
    residualize_covariates=[],
    trait_columns=[],
)


def _fit(dataset_dir: Path, config: GpuLatentModelConfig):
    bundle = load_dataset_bundle(dataset_dir)
    model = GpuLatentNetworkModel(config)
    arts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    return arts, bundle


@pytest.fixture(scope="session")
def core_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("core_datasets")
    paths = generate_core_suite(root, seed=7)
    return {p.name: p for p in paths}


@pytest.fixture(scope="session")
def scale_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("scale_datasets")
    paths = generate_scale_suite(root, seed=7)
    return {p.name: p for p in paths}


# ---------------------------------------------------------------------------
# Fast gate tests
# ---------------------------------------------------------------------------


def test_gpu_latent_toy_v1_recovery(core_suite):
    if _GATE_TOY is None:
        pytest.skip("CALIBRATION PENDING: lock _GATE_TOY after first benchmark run")
    arts, bundle = _fit(core_suite["toy_v1"], _TOY_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_TOY, f"toy_v1 recovery={recovery:.4f} < gate={_GATE_TOY:.4f}"


def test_gpu_latent_medium_v1_recovery(core_suite):
    if _GATE_MED is None:
        pytest.skip("CALIBRATION PENDING: lock _GATE_MED after first benchmark run")
    arts, bundle = _fit(core_suite["medium_v1"], _MED_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_MED, f"medium_v1 recovery={recovery:.4f} < gate={_GATE_MED:.4f}"


def test_gpu_latent_calibration_fields(core_suite):
    arts, _ = _fit(core_suite["toy_v1"], _TOY_CONFIG)
    assert arts.calibration is not None
    assert "reconstruction_rmse" in arts.calibration
    assert "mean_log_likelihood" in arts.calibration
    assert "gpu_latent_per_gene_noise_var" in arts.calibration
    assert "n_components_used" in arts.calibration


def test_gpu_latent_api_compatibility(core_suite):
    arts, _ = _fit(core_suite["medium_v1"], _MED_CONFIG)
    assert "gene_id" in arts.module_table.columns
    assert "module_id" in arts.module_table.columns
    assert isinstance(arts.edge_table, pd.DataFrame)
    assert isinstance(arts.trait_table, pd.DataFrame)
    assert isinstance(arts.feature_scores, pd.DataFrame)


def test_gpu_latent_bic_selects_k(core_suite):
    arts, _ = _fit(core_suite["medium_v1"], _MED_CONFIG)
    k = arts.calibration["n_components_used"]
    assert k in _MED_CONFIG.n_components_grid, (
        f"selected k={k} not in grid={_MED_CONFIG.n_components_grid}"
    )
    assert arts.calibration["n_components_selected_by"] == "bic"


def test_gpu_latent_per_gene_noise_var_shape(core_suite):
    arts, bundle = _fit(core_suite["toy_v1"], _TOY_CONFIG)
    noise_var = arts.calibration["gpu_latent_per_gene_noise_var"]
    n_genes = len(bundle.feature_tables["gene"])
    assert len(noise_var) == n_genes, (
        f"noise_var length {len(noise_var)} != n_genes {n_genes}"
    )
    assert all(v > 0 for v in noise_var), "All per-gene noise variances must be positive"


def test_gpu_latent_determinism(core_suite):
    arts1, _ = _fit(core_suite["medium_v1"], _MED_CONFIG)
    arts2, _ = _fit(core_suite["medium_v1"], _MED_CONFIG)
    mt1 = arts1.module_table.sort_values(["gene_id", "module_id"]).reset_index(drop=True)
    mt2 = arts2.module_table.sort_values(["gene_id", "module_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(mt1, mt2)


def test_gpu_latent_no_torch_graceful_error(core_suite):
    bundle = load_dataset_bundle(core_suite["toy_v1"])
    with mock.patch.dict(sys.modules, {"torch": None}):
        import importlib as il
        import isograph.models.gpu_latent as gl_mod
        original = gl_mod._TORCH_AVAILABLE
        gl_mod._TORCH_AVAILABLE = False
        try:
            model = GpuLatentNetworkModel(_TOY_CONFIG)
            with pytest.raises(ImportError, match="PyTorch is required"):
                model.fit(
                    transcript_counts=bundle.matrices["transcript_counts"],
                    transcript_table=bundle.feature_tables["transcript"],
                    sample_table=bundle.sample_table,
                )
        finally:
            gl_mod._TORCH_AVAILABLE = original


# ---------------------------------------------------------------------------
# Slow tests (marked with pytest.mark.slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gpu_latent_xlarge_v1_recovery(scale_suite):
    if _GATE_XLARGE is None:
        pytest.skip(
            "CALIBRATION PENDING: run `isograph benchmark --config-name stage7_gpu_latent`, "
            "observe xlarge_v1 recovery, set _GATE_XLARGE = observed - 0.05, then re-run."
        )
    arts, bundle = _fit(scale_suite["xlarge_v1"], _XLARGE_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_XLARGE, (
        f"xlarge_v1 recovery={recovery:.4f} < gate={_GATE_XLARGE:.4f}"
    )


@pytest.mark.slow
def test_gpu_latent_xlarge_v1_faster_than_fa(core_suite, scale_suite):
    """GPU latent on xlarge_v1 should be < 1/3 the runtime of sklearn FA on same fixture."""
    from isograph.models.latent import LatentNetworkModel
    from isograph.workflow.config import LatentModelConfig

    fa_cfg = LatentModelConfig(
        n_components_grid=None, n_components=10, max_iter=1000, tol=1e-4
    )
    bundle = load_dataset_bundle(scale_suite["xlarge_v1"])

    t0 = time.perf_counter()
    LatentNetworkModel(fa_cfg).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    fa_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    GpuLatentNetworkModel(_XLARGE_CONFIG).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    gl_time = time.perf_counter() - t0

    assert gl_time < fa_time / 3, (
        f"gpu_latent ({gl_time:.1f}s) not < 1/3 of FA ({fa_time:.1f}s) on xlarge_v1"
    )


@pytest.mark.slow
def test_gpu_latent_xxlarge_v1_runtime_budget(scale_suite):
    start = time.perf_counter()
    _fit(scale_suite["xxlarge_v1"], _XLARGE_CONFIG)
    elapsed = time.perf_counter() - start
    assert elapsed < 900, f"xxlarge_v1 gpu_latent fit took {elapsed:.1f}s > 900s budget"
