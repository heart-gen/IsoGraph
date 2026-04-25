"""Stage 4 promotion gate tests.

Covers Stage 4 checklist items:
  - toy_v1 recovery == 1.0 with vae backend
  - medium_v1 recovery >= 1.0 with vae backend
  - nonlinear_v1 recovery >= 0.750 (promotion gate)
  - Calibration fields present
  - FitArtifacts schema valid
  - medium_v1 snapshot determinism
  - Seed sensitivity on toy_v1
  - Checkpoint save/load round-trip
  - Single-isoform gene handling
  - Graceful ImportError when torch absent
  - Benchmark runner integration (slow)
  - nonlinear_v1 multi-seed recovery (slow)
  - Runtime budget on medium_v1 (slow)
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

pytest.importorskip("torch", reason="PyTorch not installed — skip VAE gate tests")

from isograph.benchmarks.synthetic import generate_core_suite
from isograph.evaluation.metrics import module_recovery_score
from isograph.evaluation.runner import benchmark
from isograph.evaluation.snapshots import compare_snapshot_dirs, save_snapshot
from isograph.io.artifacts import (
    DatasetBundle,
    build_feature_spec,
    build_matrix_spec,
    load_dataset_bundle,
)
from isograph.models.vae import VaeNetworkModel
from isograph.validation import DatasetManifest
from isograph.workflow.config import BenchmarkCommandConfig, VaeModelConfig

_TOY_CONFIG = VaeModelConfig(
    latent_dim=4, hidden_dim=64, n_epochs=500, beta=0.5, alpha=0.70,
    min_module_size=2, random_state=7,
)
_MEDIUM_CONFIG = VaeModelConfig(
    latent_dim=12, hidden_dim=128, n_epochs=500, beta=0.5, alpha=0.70,
    min_module_size=2, random_state=7,
)
_NONLINEAR_CONFIG = VaeModelConfig(
    latent_dim=4, hidden_dim=128, n_epochs=800, beta=0.5, alpha=0.80,
    min_module_size=2, random_state=7,
)

_REQUIRED_CALIBRATION_KEYS = {
    "reconstruction_rmse",
    "vae_final_elbo",
    "vae_final_recon_loss",
    "vae_final_kl_loss",
    "vae_best_epoch",
    "vae_early_stopped",
    "vae_n_epochs_trained",
    "vae_beta_used",
    "vae_latent_dim",
    "converged",
    "vae_per_dim_kl",
    "vae_n_collapsed_dims",
    "vae_collapsed_dim_indices",
    "vae_mean_kl",
    "vae_posterior_collapse",
}


def _fit_vae(dataset_dir: Path, config: VaeModelConfig):
    bundle = load_dataset_bundle(dataset_dir)
    model = VaeNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    return artifacts, bundle


def _minimal_bundle(n_genes, n_samples, transcript_counts, transcript_table, gene_ids):
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    sample_table = pd.DataFrame({"sample_id": sample_ids})
    gene_table = pd.DataFrame({"gene_id": gene_ids})
    gene_counts = np.ones((n_genes, n_samples), dtype=float)
    psi_table = pd.DataFrame({"psi_uid": gene_ids, "gene_id": gene_ids})
    psi = np.full((n_genes, n_samples), 0.5)
    manifest = DatasetManifest(
        dataset_name="test_minimal",
        suite_name="test",
        description="minimal test fixture",
        sample_table="samples.parquet",
        feature_tables=[
            build_feature_spec("transcript", "transcripts.parquet", transcript_table),
            build_feature_spec("gene", "genes.parquet", gene_table),
            build_feature_spec("psi", "psi.parquet", psi_table),
        ],
        matrices=[
            build_matrix_spec("transcript_counts", "transcript_counts.npz", transcript_counts),
            build_matrix_spec("gene_counts", "gene_counts.npz", gene_counts),
            build_matrix_spec("psi", "psi.npz", psi),
        ],
        provenance={"generator": "test"},
    )
    return DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables={"transcript": transcript_table, "gene": gene_table, "psi": psi_table},
        matrices={"transcript_counts": transcript_counts, "gene_counts": gene_counts, "psi": psi},
        truth_tables={},
    )


@pytest.fixture(scope="session")
def core_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets")
    paths = generate_core_suite(root, seed=7)
    return {p.name: p for p in paths}


# ---------------------------------------------------------------------------
# Fast gate tests
# ---------------------------------------------------------------------------


def test_vae_toy_v1_recovery(core_suite):
    arts, bundle = _fit_vae(core_suite["toy_v1"], _TOY_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery == 1.0, f"toy_v1 recovery={recovery:.4f}, expected 1.0"


def test_vae_medium_v1_recovery(core_suite):
    arts, bundle = _fit_vae(core_suite["medium_v1"], _MEDIUM_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= 1.0, f"medium_v1 recovery={recovery:.4f}, expected >= 1.0"


def test_vae_nonlinear_v1_recovery(core_suite):
    arts, bundle = _fit_vae(core_suite["nonlinear_v1"], _NONLINEAR_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= 0.750, f"nonlinear_v1 recovery={recovery:.4f} < 0.750 (promotion gate)"


def test_vae_calibration_fields_present(core_suite):
    arts, _ = _fit_vae(core_suite["toy_v1"], _TOY_CONFIG)
    assert arts.calibration is not None
    missing = _REQUIRED_CALIBRATION_KEYS - set(arts.calibration.keys())
    assert not missing, f"Missing calibration keys: {missing}"


def test_vae_api_compatibility(core_suite):
    arts, _ = _fit_vae(core_suite["toy_v1"], _TOY_CONFIG)
    assert isinstance(arts.module_table, pd.DataFrame)
    assert isinstance(arts.edge_table, pd.DataFrame)
    assert isinstance(arts.trait_table, pd.DataFrame)
    assert isinstance(arts.feature_scores, pd.DataFrame)
    assert arts.calibration is not None
    # checkpoint_path is None unless checkpoint_dir is set
    assert arts.checkpoint_path is None


def test_vae_determinism_medium_v1(core_suite, tmp_path):
    arts1, bundle = _fit_vae(core_suite["medium_v1"], _MEDIUM_CONFIG)
    arts2, _ = _fit_vae(core_suite["medium_v1"], _MEDIUM_CONFIG)
    snap1 = tmp_path / "snap1"
    snap2 = tmp_path / "snap2"
    snap1.mkdir(); snap2.mkdir()
    model_config = _MEDIUM_CONFIG
    metrics = {"n_modules": 0, "n_edges": 0, "recovery": None, "runtime_seconds": 0.0}
    save_snapshot(arts1, model_config, metrics, snap1, "det_v1", "medium_v1")
    save_snapshot(arts2, model_config, metrics, snap2, "det_v1", "medium_v1")
    report = compare_snapshot_dirs(snap1, snap2)
    assert report["passed"], (
        "medium_v1 VAE snapshots are not deterministic:\n" + "\n".join(report["differences"])
    )


def test_vae_seed_sensitivity_toy_v1(core_suite):
    bundle = load_dataset_bundle(core_suite["toy_v1"])
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recoveries = []
    for seed in range(5):
        cfg = VaeModelConfig(latent_dim=4, hidden_dim=64, n_epochs=500, beta=0.5, alpha=0.70,
                             min_module_size=2, random_state=seed)
        model = VaeNetworkModel(cfg)
        arts = model.fit(
            transcript_counts=bundle.matrices["transcript_counts"],
            transcript_table=bundle.feature_tables["transcript"],
            sample_table=bundle.sample_table,
        )
        recoveries.append(module_recovery_score(arts.module_table, truth))
    assert np.std(recoveries) < 0.05, f"Seed sensitivity too high: std={np.std(recoveries):.4f}"
    assert all(r >= 0.95 for r in recoveries), f"Some seeds below 0.95: {recoveries}"


def test_vae_checkpoint_save_load(core_suite, tmp_path):
    from isograph.models.vae import load_vae_checkpoint
    cfg = VaeModelConfig(latent_dim=4, hidden_dim=32, n_epochs=200, beta=0.5, alpha=0.70,
                         min_module_size=2, random_state=7, checkpoint_dir=tmp_path)
    bundle = load_dataset_bundle(core_suite["toy_v1"])
    model = VaeNetworkModel(cfg)
    model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    chk_path = tmp_path / "vae_checkpoint.pt"
    assert chk_path.exists(), "Checkpoint file not saved"

    import torch
    data = torch.load(chk_path, map_location="cpu", weights_only=True)
    n_genes = data["n_genes"]
    loaded = load_vae_checkpoint(chk_path, cfg, n_genes)
    assert loaded is not None


def test_vae_single_isoform_handling():
    n_genes, n_samples = 1, 20
    gene_ids = ["G0000"]
    tx_ids = ["G0000_tx1", "G0000_tx2"]
    transcript_table = pd.DataFrame(
        {"transcript_id": tx_ids, "gene_id": [gene_ids[0], gene_ids[0]],
         "chrom": ["chr1", "chr1"], "start": [0, 0], "end": [100, 100]}
    )
    rng = np.random.default_rng(0)
    transcript_counts = rng.integers(0, 10, size=(2, n_samples)).astype(float)
    bundle = _minimal_bundle(n_genes, n_samples, transcript_counts, transcript_table, gene_ids)
    cfg = VaeModelConfig(latent_dim=2, hidden_dim=16, n_epochs=10, alpha=0.70, random_state=0)
    model = VaeNetworkModel(cfg)
    arts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    assert isinstance(arts.module_table, pd.DataFrame)


def test_vae_latent_dim_grid_selects_k(core_suite):
    """Grid sweep selects a latent_dim that achieves at least baseline recovery."""
    cfg = VaeModelConfig(
        latent_dim_grid=[2, 4, 8], hidden_dim=64, n_epochs=500, beta=0.5,
        alpha=0.70, min_module_size=2, random_state=7,
    )
    arts, bundle = _fit_vae(core_suite["toy_v1"], cfg)
    assert arts.calibration is not None
    assert "latent_dim_selected" in arts.calibration, "latent_dim_selected missing from calibration"
    assert "latent_dim_grid_rmses" in arts.calibration, "latent_dim_grid_rmses missing from calibration"
    assert arts.calibration["latent_dim_selected"] in [2, 4, 8]
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= 0.95, f"grid sweep toy_v1 recovery={recovery:.4f} < 0.95"


def test_vae_no_torch_graceful_error(core_suite):
    """When torch is absent, fit() should raise ImportError with a clear message."""
    import isograph.models.vae as vae_module
    original = vae_module._TORCH_AVAILABLE
    try:
        vae_module._TORCH_AVAILABLE = False
        bundle = load_dataset_bundle(core_suite["toy_v1"])
        model = VaeNetworkModel(_TOY_CONFIG)
        with pytest.raises(ImportError, match="PyTorch"):
            model.fit(
                transcript_counts=bundle.matrices["transcript_counts"],
                transcript_table=bundle.feature_tables["transcript"],
                sample_table=bundle.sample_table,
            )
    finally:
        vae_module._TORCH_AVAILABLE = original


# ---------------------------------------------------------------------------
# Slow tests (marked with pytest.mark.slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_vae_benchmark_runner(core_suite, tmp_path):
    cfg = BenchmarkCommandConfig(
        command="benchmark",
        dataset_suite="core_v1",
        stage_name="stage4_vae_test",
        backend="vae",
        fixture_filter="toy_v1",
        benchmark_root=tmp_path / "benchmarks",
        artifacts_root=tmp_path / "artifacts",
        dataset_root=core_suite["toy_v1"].parent,
        report_root=tmp_path / "reports",
        seed=7,
        vae=_TOY_CONFIG,
        fixture_vae_overrides={},
        recovery_thresholds={"toy_v1": 1.0},
    )
    result = benchmark(cfg)
    assert "report" in result
    import json
    report = json.loads(result["report"].read_text())
    rows = {r["dataset"]: r for r in report["results"]}
    assert rows["toy_v1"]["recovery"] == 1.0
    assert not report["gate_failures"]


@pytest.mark.slow
def test_vae_nonlinear_v1_multi_seed(core_suite):
    bundle = load_dataset_bundle(core_suite["nonlinear_v1"])
    truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
    recoveries = []
    for seed in [0, 1, 2]:
        cfg = VaeModelConfig(latent_dim=4, hidden_dim=128, n_epochs=800, beta=0.5, alpha=0.80,
                             min_module_size=2, random_state=seed)
        model = VaeNetworkModel(cfg)
        arts = model.fit(
            transcript_counts=bundle.matrices["transcript_counts"],
            transcript_table=bundle.feature_tables["transcript"],
            sample_table=bundle.sample_table,
        )
        recoveries.append(module_recovery_score(arts.module_table, truth))
    assert np.mean(recoveries) >= 0.750, f"mean recovery={np.mean(recoveries):.4f} < 0.750"


@pytest.mark.slow
def test_vae_runtime_budget_medium_v1(core_suite):
    start = time.perf_counter()
    _fit_vae(core_suite["medium_v1"], _MEDIUM_CONFIG)
    elapsed = time.perf_counter() - start
    assert elapsed < 600, f"medium_v1 fit took {elapsed:.1f}s > 600s budget"
