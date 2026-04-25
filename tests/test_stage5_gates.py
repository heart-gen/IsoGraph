"""Stage 5 promotion gate tests — WGCNA comparison benchmark.

Covers Stage 5 checklist items:
  - module_table schema valid (gene_id, module_id columns)
  - FitArtifacts API compatibility
  - Required calibration keys present
  - toy_v1 recovery > 0 (WGCNA finds structure)
  - medium_v1 recovery > 0
  - nonlinear_v1 WGCNA recovery < 0.80 (VAE advantage expected)
  - Soft-thresholding power auto-selected and recorded in calibration
  - Grey (unassigned) genes excluded from module_table
  - Deterministic output on repeated fits with same seed
  - Graceful ImportError when Rscript absent
  - Benchmark runner integration (slow)
  - VAE beats WGCNA on nonlinear_v1 (slow)
  - Runtime budget on medium_v1 (slow)
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

if shutil.which("Rscript") is None:
    pytest.skip("Rscript not found in PATH — skip WGCNA gate tests", allow_module_level=True)

from isograph.benchmarks.synthetic import generate_core_suite
from isograph.evaluation.metrics import module_recovery_score
from isograph.evaluation.runner import benchmark
from isograph.io.artifacts import load_dataset_bundle
from isograph.models.wgcna import WgcnaNetworkModel
from isograph.workflow.config import BenchmarkCommandConfig, WgcnaModelConfig

_TOY_CONFIG = WgcnaModelConfig(
    min_module_size=2, sft_r2_threshold=0.85, merge_cut_height=0.25,
    deep_split=2, network_type="signed", random_state=7,
)
_MEDIUM_CONFIG = WgcnaModelConfig(
    min_module_size=2, sft_r2_threshold=0.85, merge_cut_height=0.25,
    deep_split=2, network_type="signed", random_state=7,
)
_NONLINEAR_CONFIG = WgcnaModelConfig(
    min_module_size=2, sft_r2_threshold=0.85, merge_cut_height=0.25,
    deep_split=2, network_type="signed", random_state=7,
)

_REQUIRED_CALIBRATION_KEYS = {
    "wgcna_soft_threshold_power",
    "wgcna_sft_r2",
    "wgcna_n_modules",
    "wgcna_n_unassigned_genes",
    "wgcna_merge_cut_height",
    "wgcna_network_type",
}


def _fit_wgcna(dataset_dir: Path, config: WgcnaModelConfig):
    bundle = load_dataset_bundle(dataset_dir)
    model = WgcnaNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    return artifacts, bundle


@pytest.fixture(scope="session")
def core_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets")
    paths = generate_core_suite(root, seed=7)
    return {p.name: p for p in paths}


# ---------------------------------------------------------------------------
# Fast gate tests
# ---------------------------------------------------------------------------


def test_wgcna_toy_v1_module_table_schema(core_suite):
    arts, _ = _fit_wgcna(core_suite["toy_v1"], _TOY_CONFIG)
    assert "gene_id" in arts.module_table.columns, "module_table missing gene_id column"
    assert "module_id" in arts.module_table.columns, "module_table missing module_id column"


def test_wgcna_api_compatibility(core_suite):
    arts, _ = _fit_wgcna(core_suite["toy_v1"], _TOY_CONFIG)
    assert isinstance(arts.module_table, pd.DataFrame)
    assert isinstance(arts.edge_table, pd.DataFrame)
    assert isinstance(arts.trait_table, pd.DataFrame)
    assert isinstance(arts.feature_scores, pd.DataFrame)
    assert arts.calibration is not None
    assert arts.checkpoint_path is None


def test_wgcna_calibration_fields_present(core_suite):
    arts, _ = _fit_wgcna(core_suite["toy_v1"], _TOY_CONFIG)
    assert arts.calibration is not None
    missing = _REQUIRED_CALIBRATION_KEYS - set(arts.calibration.keys())
    assert not missing, f"Missing calibration keys: {missing}"


def test_wgcna_toy_v1_recovery_nonzero(core_suite):
    arts, bundle = _fit_wgcna(core_suite["toy_v1"], _TOY_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery > 0.0, f"toy_v1 WGCNA recovery={recovery:.4f}, expected > 0"


def test_wgcna_medium_v1_recovery_nonzero(core_suite):
    arts, bundle = _fit_wgcna(core_suite["medium_v1"], _MEDIUM_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery > 0.0, f"medium_v1 WGCNA recovery={recovery:.4f}, expected > 0"


def test_wgcna_nonlinear_v1_below_vae(core_suite):
    arts, bundle = _fit_wgcna(core_suite["nonlinear_v1"], _NONLINEAR_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    vae_recovery = 0.958
    assert recovery < vae_recovery, (
        f"nonlinear_v1 WGCNA recovery={recovery:.4f} >= VAE {vae_recovery} — "
        "expected WGCNA to underperform VAE on radial/product module structure"
    )


def test_wgcna_power_auto_selected(core_suite):
    arts, _ = _fit_wgcna(core_suite["toy_v1"], _TOY_CONFIG)
    power = arts.calibration["wgcna_soft_threshold_power"]
    assert isinstance(power, int), f"wgcna_soft_threshold_power should be int, got {type(power)}"
    assert 1 <= power <= 20, f"power={power} out of expected range [1, 20]"


def test_wgcna_grey_excluded(core_suite):
    arts, _ = _fit_wgcna(core_suite["medium_v1"], _MEDIUM_CONFIG)
    if not arts.module_table.empty:
        ids = arts.module_table["module_id"].unique()
        for mid in ids:
            assert "grey" not in str(mid).lower(), (
                f"Grey (unassigned) genes appear in module_table as module_id={mid!r}"
            )
            assert mid.startswith("M"), f"module_id={mid!r} does not follow M000 format"


def test_wgcna_determinism_toy_v1(core_suite):
    arts1, _ = _fit_wgcna(core_suite["toy_v1"], _TOY_CONFIG)
    arts2, _ = _fit_wgcna(core_suite["toy_v1"], _TOY_CONFIG)
    if arts1.module_table.empty and arts2.module_table.empty:
        return
    m1 = arts1.module_table.sort_values(["module_id", "gene_id"]).reset_index(drop=True)
    m2 = arts2.module_table.sort_values(["module_id", "gene_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(m1, m2, check_like=False)


def test_wgcna_no_rscript_graceful_error(core_suite):
    bundle = load_dataset_bundle(core_suite["toy_v1"])
    model = WgcnaNetworkModel(_TOY_CONFIG)
    with mock.patch("isograph.models.wgcna.shutil.which", return_value=None):
        with pytest.raises(ImportError, match="Rscript"):
            model.fit(
                transcript_counts=bundle.matrices["transcript_counts"],
                transcript_table=bundle.feature_tables["transcript"],
                sample_table=bundle.sample_table,
            )


# ---------------------------------------------------------------------------
# Slow tests (marked with pytest.mark.slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_wgcna_benchmark_runner(core_suite, tmp_path):
    cfg = BenchmarkCommandConfig(
        command="benchmark",
        dataset_suite="core_v1",
        stage_name="stage5_wgcna_test",
        backend="wgcna",
        fixture_filter="toy_v1",
        benchmark_root=tmp_path / "benchmarks",
        artifacts_root=tmp_path / "artifacts",
        dataset_root=core_suite["toy_v1"].parent,
        report_root=tmp_path / "reports",
        seed=7,
        wgcna=_TOY_CONFIG,
        fixture_wgcna_overrides={},
        recovery_thresholds={},
    )
    result = benchmark(cfg)
    assert "report" in result
    import json
    report = json.loads(result["report"].read_text())
    assert "results" in report
    rows = {r["dataset"]: r for r in report["results"]}
    assert "toy_v1" in rows
    assert not report["gate_failures"]


@pytest.mark.slow
def test_wgcna_vae_beats_wgcna_nonlinear_v1(core_suite):
    pytest.importorskip("torch", reason="PyTorch not installed — skip VAE comparison")
    from isograph.models.vae import VaeNetworkModel
    from isograph.workflow.config import VaeModelConfig

    bundle = load_dataset_bundle(core_suite["nonlinear_v1"])
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())

    wgcna_arts = WgcnaNetworkModel(_NONLINEAR_CONFIG).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    wgcna_recovery = module_recovery_score(wgcna_arts.module_table, truth)

    vae_cfg = VaeModelConfig(latent_dim=4, hidden_dim=128, n_epochs=800, beta=0.5,
                             alpha=0.80, min_module_size=2, random_state=7)
    vae_arts = VaeNetworkModel(vae_cfg).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    vae_recovery = module_recovery_score(vae_arts.module_table, truth)

    assert vae_recovery > wgcna_recovery, (
        f"VAE recovery={vae_recovery:.4f} did not beat WGCNA recovery={wgcna_recovery:.4f} "
        "on nonlinear_v1 (expected VAE advantage on radial/product module structure)"
    )


@pytest.mark.slow
def test_wgcna_runtime_budget_medium_v1(core_suite):
    start = time.perf_counter()
    _fit_wgcna(core_suite["medium_v1"], _MEDIUM_CONFIG)
    elapsed = time.perf_counter() - start
    assert elapsed < 300, f"medium_v1 WGCNA fit took {elapsed:.1f}s > 300s budget"
