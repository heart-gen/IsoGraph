"""Stage 6 promotion gate tests — large-scale VAE architecture scaling.

Covers Stage 6 checklist items:
  - xlarge_v1 fixture (6k genes / 240 samples, 25:1) loads correctly
  - xxlarge_v1 fixture (12k genes / 240 samples, 50:1) loads correctly
  - module_table schema valid for both fixtures
  - VAE calibration fields present
  - Auto batch_size kicks in when n_genes > 2000 and batch_size=None
  - Default backend is now "vae"
  - VAE recovery on xlarge_v1 >= GATE_XLARGE  (slow; gate set post-calibration)
  - VAE recovery on xxlarge_v1 >= GATE_XXLARGE (slow; gate set post-calibration)
  - Runtime within budget (slow)

CALIBRATION PROCESS
-------------------
1. Run: isograph benchmark --config-name stage6_vae_xlarge
2. Check artifacts/reports/stage6_vae_xlarge-benchmark.json for observed recovery.
3. Set _GATE_XLARGE  = observed_xlarge  - 0.10
   Set _GATE_XXLARGE = observed_xxlarge - 0.10
4. Commit the updated constants and re-run this file.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="PyTorch not installed — skip Stage 6 VAE gate tests")

from isograph.benchmarks.synthetic import generate_scale_suite
from isograph.evaluation.metrics import module_recovery_score
from isograph.io.artifacts import load_dataset_bundle
from isograph.models.vae import VaeNetworkModel
from isograph.workflow.config import BenchmarkCommandConfig, VaeModelConfig

# ---------------------------------------------------------------------------
# Calibration-locked recovery gates.
# Set to None until the first calibration benchmark run completes.
# After calibration: set to observed_recovery - 0.10.
# ---------------------------------------------------------------------------
_GATE_XLARGE: float | None = 0.90    # observed=1.0 − 0.10; seed=7; 2026-04-23
_GATE_XXLARGE: float | None = 0.90   # observed=1.0 − 0.10; seed=7; 2026-04-23
_GATE_STRESS: float | None = 0.90    # observed=1.0 − 0.10; seed=7; 2026-04-25

_XLARGE_CONFIG = VaeModelConfig(
    hidden_dim=512, n_hidden_layers=3, batch_size=64,
    n_epochs=800, beta=0.5, alpha=0.75, min_module_size=2, random_state=7,
)
_XXLARGE_CONFIG = VaeModelConfig(
    hidden_dim=512, n_hidden_layers=3, batch_size=64,
    n_epochs=1000, beta=0.5, alpha=0.75, min_module_size=2, random_state=7,
)
_STRESS_CONFIG = VaeModelConfig(
    hidden_dim=512, n_hidden_layers=3, batch_size=64,
    n_epochs=1600, beta=0.5, alpha=0.75, min_module_size=2, random_state=7,
)


def _fit_vae(dataset_dir: Path, config: VaeModelConfig):
    bundle = load_dataset_bundle(dataset_dir)
    model = VaeNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    return artifacts, bundle


@pytest.fixture(scope="session")
def scale_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("scale_datasets")
    paths = generate_scale_suite(root, seed=7)
    return {p.name: p for p in paths}


# ---------------------------------------------------------------------------
# Fast gate tests
# ---------------------------------------------------------------------------


def test_xlarge_v1_fixture_schema(scale_suite):
    bundle = load_dataset_bundle(scale_suite["xlarge_v1"])
    assert "transcript_counts" in bundle.matrices
    assert "truth_modules.parquet" in bundle.truth_tables
    assert len(bundle.feature_tables["gene"]) == 6000
    assert len(bundle.sample_table) == 240


def test_xxlarge_v1_fixture_schema(scale_suite):
    bundle = load_dataset_bundle(scale_suite["xxlarge_v1"])
    assert "transcript_counts" in bundle.matrices
    assert "truth_modules.parquet" in bundle.truth_tables
    assert len(bundle.feature_tables["gene"]) == 12000
    assert len(bundle.sample_table) == 240


def test_xlarge_v1_vae_module_table_schema(scale_suite):
    arts, _ = _fit_vae(scale_suite["xlarge_v1"], _XLARGE_CONFIG)
    assert "gene_id" in arts.module_table.columns
    assert "module_id" in arts.module_table.columns


def test_xxlarge_v1_vae_module_table_schema(scale_suite):
    arts, _ = _fit_vae(scale_suite["xxlarge_v1"], _XXLARGE_CONFIG)
    assert "gene_id" in arts.module_table.columns
    assert "module_id" in arts.module_table.columns


def test_vae_auto_batch_size_large_input(scale_suite):
    """When n_genes > 2000 and batch_size=None, VAE auto-sets an effective batch size."""
    cfg_no_batch = VaeModelConfig(
        hidden_dim=512, n_hidden_layers=3, batch_size=None,
        n_epochs=5, beta=0.5, alpha=0.75, random_state=7,
    )
    bundle = load_dataset_bundle(scale_suite["xlarge_v1"])
    model = VaeNetworkModel(cfg_no_batch)
    arts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    assert isinstance(arts.module_table, pd.DataFrame)


def test_vae_xlarge_calibration_fields(scale_suite):
    arts, _ = _fit_vae(scale_suite["xlarge_v1"],
                       VaeModelConfig(hidden_dim=512, n_hidden_layers=3, batch_size=64,
                                      n_epochs=5, alpha=0.75, random_state=7))
    assert arts.calibration is not None
    assert "reconstruction_rmse" in arts.calibration
    assert "vae_latent_dim" in arts.calibration


def test_vae_xxlarge_calibration_fields(scale_suite):
    arts, _ = _fit_vae(scale_suite["xxlarge_v1"],
                       VaeModelConfig(hidden_dim=512, n_hidden_layers=3, batch_size=64,
                                      n_epochs=5, alpha=0.75, random_state=7))
    assert arts.calibration is not None
    assert "reconstruction_rmse" in arts.calibration


def test_vae_default_backend_is_vae():
    assert BenchmarkCommandConfig().backend == "vae"


def test_xxlarge_stress_v1_fixture_schema(scale_suite):
    bundle = load_dataset_bundle(scale_suite["xxlarge_stress_v1"])
    assert "transcript_counts" in bundle.matrices
    assert "truth_modules.parquet" in bundle.truth_tables
    assert len(bundle.feature_tables["gene"]) == 12000
    assert len(bundle.sample_table) == 240


def test_xxlarge_stress_v1_vae_module_table_schema(scale_suite):
    arts, _ = _fit_vae(scale_suite["xxlarge_stress_v1"], _STRESS_CONFIG)
    assert "gene_id" in arts.module_table.columns
    assert "module_id" in arts.module_table.columns


def test_vae_xxlarge_stress_calibration_fields(scale_suite):
    arts, _ = _fit_vae(scale_suite["xxlarge_stress_v1"],
                       VaeModelConfig(hidden_dim=512, n_hidden_layers=3, batch_size=64,
                                      n_epochs=5, alpha=0.75, random_state=7))
    assert arts.calibration is not None
    assert "reconstruction_rmse" in arts.calibration


# ---------------------------------------------------------------------------
# Slow tests (marked with pytest.mark.slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_xlarge_v1_vae_recovery(scale_suite):
    if _GATE_XLARGE is None:
        pytest.skip(
            "CALIBRATION PENDING: run `isograph benchmark --config-name stage6_vae_xlarge`, "
            "observe xlarge_v1 recovery, set _GATE_XLARGE = observed - 0.10, then re-run."
        )
    arts, bundle = _fit_vae(scale_suite["xlarge_v1"], _XLARGE_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_XLARGE, (
        f"xlarge_v1 recovery={recovery:.4f} < gate={_GATE_XLARGE:.4f}"
    )


@pytest.mark.slow
def test_xxlarge_v1_vae_recovery(scale_suite):
    if _GATE_XXLARGE is None:
        pytest.skip(
            "CALIBRATION PENDING: run `isograph benchmark --config-name stage6_vae_xlarge`, "
            "observe xxlarge_v1 recovery, set _GATE_XXLARGE = observed - 0.10, then re-run."
        )
    arts, bundle = _fit_vae(scale_suite["xxlarge_v1"], _XXLARGE_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_XXLARGE, (
        f"xxlarge_v1 recovery={recovery:.4f} < gate={_GATE_XXLARGE:.4f}"
    )


@pytest.mark.slow
def test_xlarge_v1_runtime_budget(scale_suite):
    start = time.perf_counter()
    _fit_vae(scale_suite["xlarge_v1"], _XLARGE_CONFIG)
    elapsed = time.perf_counter() - start
    assert elapsed < 900, f"xlarge_v1 fit took {elapsed:.1f}s > 900s budget"


@pytest.mark.slow
def test_xxlarge_v1_runtime_budget(scale_suite):
    start = time.perf_counter()
    _fit_vae(scale_suite["xxlarge_v1"], _XXLARGE_CONFIG)
    elapsed = time.perf_counter() - start
    assert elapsed < 1800, f"xxlarge_v1 fit took {elapsed:.1f}s > 1800s budget"


@pytest.mark.slow
def test_xxlarge_stress_v1_vae_recovery(scale_suite):
    if _GATE_STRESS is None:
        pytest.skip(
            "CALIBRATION PENDING: run `isograph benchmark --config-name stage6_scale_comparison_vae`, "
            "observe xxlarge_stress_v1 recovery, set _GATE_STRESS = observed - 0.10, then re-run."
        )
    arts, bundle = _fit_vae(scale_suite["xxlarge_stress_v1"], _STRESS_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_STRESS, (
        f"xxlarge_stress_v1 recovery={recovery:.4f} < gate={_GATE_STRESS:.4f}"
    )


@pytest.mark.slow
def test_xxlarge_stress_v1_runtime_budget(scale_suite):
    start = time.perf_counter()
    _fit_vae(scale_suite["xxlarge_stress_v1"], _STRESS_CONFIG)
    elapsed = time.perf_counter() - start
    assert elapsed < 2400, f"xxlarge_stress_v1 fit took {elapsed:.1f}s > 2400s budget"
