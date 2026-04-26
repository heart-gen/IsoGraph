"""Stage 3 promotion gate tests.

Covers Stage 3 checklist items:
  - toy_v1 recovery == 1.0 with graph backend
  - medium_v1 recovery >= 1.0 with graph backend
  - gamma=0.0 produces output identical to LatentNetworkModel (regression guard)
  - Calibration fields present (all Stage 2 keys + new graph keys)
  - Public FitArtifacts API unchanged
  - Single-isoform gene handling
  - medium_v1 snapshot determinism
  - Ablation: empty_graph == no_smoothing
  - Prior edges enriched within true modules (interpretability gate)
  - Graph ablation report structure
  - Benchmark runner integration (slow)
  - Runtime budget on medium_v1 (slow)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.benchmarks.synthetic import generate_core_suite
from isograph.evaluation.graph_diagnostics import (
    graph_ablation_report,
    graph_prior_edge_report,
)
from isograph.evaluation.metrics import module_recovery_score
from isograph.evaluation.runner import benchmark
from isograph.evaluation.snapshots import compare_snapshot_dirs, save_snapshot
from isograph.features.graph import build_gene_graph
from isograph.io.artifacts import (
    DatasetBundle,
    build_feature_spec,
    build_matrix_spec,
    load_dataset_bundle,
)
from isograph.models.graph import GraphNetworkModel
from isograph.models.latent import LatentNetworkModel
from isograph.validation import DatasetManifest
from isograph.workflow.config import (
    BenchmarkCommandConfig,
    GraphModelConfig,
    LatentModelConfig,
)

_TOY_CONFIG = GraphModelConfig(alpha=0.05, min_module_size=2, gamma=0.5, corr_threshold=0.3)
_MEDIUM_CONFIG = GraphModelConfig(alpha=0.01, min_module_size=2, gamma=0.5, corr_threshold=0.3)


def _fit_graph(dataset_dir: Path, config: GraphModelConfig):
    bundle = load_dataset_bundle(dataset_dir)
    model = GraphNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    return artifacts, bundle


def _minimal_bundle(
    n_genes: int,
    n_samples: int,
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    gene_ids: list[str],
) -> DatasetBundle:
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
            build_feature_spec("gene", "genes.parquet", gene_table),
            build_feature_spec("transcript", "transcripts.parquet", transcript_table),
            build_feature_spec("psi", "psi.parquet", psi_table),
        ],
        matrices=[
            build_matrix_spec("gene_counts", "gene_counts.npz", gene_counts),
            build_matrix_spec("transcript_counts", "transcript_counts.npz", transcript_counts),
            build_matrix_spec("psi", "psi.npz", psi),
        ],
        provenance={"generator": "test"},
    )
    return DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables={"gene": gene_table, "transcript": transcript_table, "psi": psi_table},
        matrices={"gene_counts": gene_counts, "transcript_counts": transcript_counts, "psi": psi},
        truth_tables={},
    )


# ---------------------------------------------------------------------------
# toy_v1 recovery gate
# ---------------------------------------------------------------------------


def test_graph_toy_v1_recovery(tmp_path: Path) -> None:
    """Graph backend recovers all modules on toy_v1 (recovery == 1.0)."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    toy_dir = next(p for p in paths if "toy" in p.name)

    artifacts, bundle = _fit_graph(toy_dir, _TOY_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet")
    assert truth is not None and not truth.empty

    recovery = module_recovery_score(artifacts.module_table, truth)
    assert recovery == 1.0, f"toy_v1 graph recovery {recovery:.4f} < Stage 3 gate 1.0"


# ---------------------------------------------------------------------------
# medium_v1 recovery gate
# ---------------------------------------------------------------------------


def test_graph_medium_v1_recovery(tmp_path: Path) -> None:
    """Graph backend matches Stage 2 on medium_v1 (recovery >= 1.0, no regression)."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    medium_dir = next(p for p in paths if "medium" in p.name)

    artifacts, bundle = _fit_graph(medium_dir, _MEDIUM_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet")
    assert truth is not None and not truth.empty

    recovery = module_recovery_score(artifacts.module_table, truth)
    assert recovery >= 1.0, f"medium_v1 graph recovery {recovery:.4f} < Stage 3 gate 1.0"


# ---------------------------------------------------------------------------
# gamma=0 regression guard
# ---------------------------------------------------------------------------


def test_graph_gamma_zero_matches_latent(tmp_path: Path) -> None:
    """GraphNetworkModel with gamma=0 produces identical feature_scores to LatentNetworkModel."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    medium_dir = next(p for p in paths if "medium" in p.name)
    bundle = load_dataset_bundle(medium_dir)

    latent_cfg = LatentModelConfig(alpha=0.01, min_module_size=2)
    graph_cfg = GraphModelConfig(
        alpha=0.01, min_module_size=2,
        gamma=0.0, edge_types=["corr"], corr_threshold=0.3,
    )

    latent_arts = LatentNetworkModel(latent_cfg).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    graph_arts = GraphNetworkModel(graph_cfg).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )

    # feature_scores is built from the original switch_matrix in both models
    latent_fs = latent_arts.feature_scores.sort_values("gene_id").reset_index(drop=True)
    graph_fs = graph_arts.feature_scores.sort_values("gene_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(latent_fs, graph_fs, check_exact=False, rtol=1e-10)

    # Module tables should agree (same FA path → same network)
    latent_mods = set(latent_arts.module_table["gene_id"].tolist())
    graph_mods = set(graph_arts.module_table["gene_id"].tolist())
    assert latent_mods == graph_mods, (
        "gamma=0 graph model assigned different genes to modules vs LatentNetworkModel"
    )


# ---------------------------------------------------------------------------
# Calibration fields
# ---------------------------------------------------------------------------


def test_graph_calibration_fields_present(tmp_path: Path) -> None:
    """FitArtifacts.calibration contains all Stage 2 keys plus new graph keys."""
    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    toy_dir = next(p for p in paths if "toy" in p.name)

    artifacts, _ = _fit_graph(toy_dir, _TOY_CONFIG)
    assert artifacts.calibration is not None, "calibration must not be None"

    stage2_keys = {
        "mean_log_likelihood", "reconstruction_rmse", "mean_noise_variance",
        "n_components_used", "n_iter", "converged", "n_components_selected_by",
    }
    graph_keys = {
        "graph_n_nodes", "graph_n_edges", "graph_mean_degree",
        "graph_edge_types_used", "graph_gamma", "graph_corr_threshold",
        "graph_smoothing_rmse",
    }
    required = stage2_keys | graph_keys
    missing = required - set(artifacts.calibration.keys())
    assert not missing, f"calibration missing keys: {missing}"

    assert artifacts.calibration["mean_log_likelihood"] is not None
    assert artifacts.calibration["reconstruction_rmse"] >= 0.0
    assert artifacts.calibration["n_components_used"] >= 1
    assert artifacts.calibration["graph_n_nodes"] > 0
    assert artifacts.calibration["graph_n_edges"] >= 0
    assert artifacts.calibration["graph_gamma"] == _TOY_CONFIG.gamma
    assert artifacts.calibration["graph_smoothing_rmse"] >= 0.0


# ---------------------------------------------------------------------------
# API compatibility
# ---------------------------------------------------------------------------


def test_graph_api_compatibility(tmp_path: Path) -> None:
    """GraphNetworkModel returns FitArtifacts with all required fields and column schemas."""
    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    toy_dir = next(p for p in paths if "toy" in p.name)

    artifacts, _ = _fit_graph(toy_dir, _TOY_CONFIG)
    assert isinstance(artifacts.module_table, pd.DataFrame)
    assert isinstance(artifacts.edge_table, pd.DataFrame)
    assert isinstance(artifacts.trait_table, pd.DataFrame)
    assert isinstance(artifacts.feature_scores, pd.DataFrame)
    assert "gene_id" in artifacts.feature_scores.columns
    if not artifacts.module_table.empty:
        assert "gene_id" in artifacts.module_table.columns
        assert "module_id" in artifacts.module_table.columns
    if not artifacts.edge_table.empty:
        assert set(["source", "target", "weight"]).issubset(artifacts.edge_table.columns)
    if not artifacts.trait_table.empty:
        assert set(["module_id", "trait", "effect", "pvalue"]).issubset(artifacts.trait_table.columns)
    assert isinstance(artifacts.eigengene_table, pd.DataFrame)
    assert "module_id" in artifacts.eigengene_table.columns


# ---------------------------------------------------------------------------
# Single-isoform gene handling
# ---------------------------------------------------------------------------


def test_graph_single_isoform_handling(tmp_path: Path) -> None:
    """A gene with only one transcript is excluded without crashing."""
    n_samples = 30
    rng = np.random.default_rng(0)

    transcript_table = pd.DataFrame(
        {
            "transcript_id": ["A_T1", "A_T2", "B_T1"],
            "gene_id": ["GeneA", "GeneA", "GeneB"],
            "length": [1000, 900, 1000],
        }
    )
    base = rng.gamma(5, 10, (3, n_samples))
    transcript_counts = np.floor(base).astype(float)

    bundle = _minimal_bundle(
        n_genes=2,
        n_samples=n_samples,
        transcript_counts=transcript_counts,
        transcript_table=transcript_table,
        gene_ids=["GeneA", "GeneB"],
    )

    model = GraphNetworkModel(GraphModelConfig(alpha=0.05, min_module_size=1, gamma=0.5))
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )

    gene_ids_in_scores = set(artifacts.feature_scores["gene_id"].tolist())
    assert "GeneB" not in gene_ids_in_scores, "Single-isoform GeneB should be excluded"
    assert "GeneA" in gene_ids_in_scores, "GeneA (2 transcripts) should be present"


# ---------------------------------------------------------------------------
# medium_v1 snapshot determinism
# ---------------------------------------------------------------------------


def test_graph_determinism_medium_v1(tmp_path: Path) -> None:
    """Two graph fits on medium_v1 produce identical snapshots."""
    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    medium_dir = next(p for p in paths if "medium" in p.name)

    snap_a = tmp_path / "snap_a"
    snap_b = tmp_path / "snap_b"

    for snap_dir in (snap_a, snap_b):
        artifacts, bundle = _fit_graph(medium_dir, _MEDIUM_CONFIG)
        truth = bundle.truth_tables.get("truth_modules.parquet")
        recovery = module_recovery_score(artifacts.module_table, truth) if truth is not None else None
        metrics = {
            "n_modules": 0 if artifacts.module_table.empty else artifacts.module_table["module_id"].nunique(),
            "n_edges": len(artifacts.edge_table),
            "recovery": recovery,
            "runtime_seconds": 0.0,
        }
        save_snapshot(
            fit_artifacts=artifacts,
            model_config=_MEDIUM_CONFIG,
            metrics=metrics,
            output_dir=snap_dir,
            snapshot_name="stage3_medium_v1_graph_v1_seed0000",
            dataset_name="medium_v1",
        )

    report = compare_snapshot_dirs(snap_a, snap_b)
    assert report["passed"], (
        "medium_v1 graph snapshots are not deterministic:\n" + "\n".join(report["differences"])
    )


# ---------------------------------------------------------------------------
# Ablation: empty_graph equals no_smoothing
# ---------------------------------------------------------------------------


def test_graph_ablation_empty_equals_no_smoothing(tmp_path: Path) -> None:
    """Ablation with edge_types=[] (empty graph) produces same result as gamma=0.0.

    Both cases have L=0 or gamma=0, so (I + gamma*L)^{-1} = I.
    """
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    medium_dir = next(p for p in paths if "medium" in p.name)
    bundle = load_dataset_bundle(medium_dir)

    # empty_graph: L is zero matrix because no edges → smoothing is identity
    cfg_empty = GraphModelConfig(
        alpha=0.01, min_module_size=2, gamma=0.5, edge_types=[]
    )
    # no_smoothing: gamma=0 → laplacian_smooth returns input unchanged
    cfg_no_smooth = GraphModelConfig(
        alpha=0.01, min_module_size=2, gamma=0.0, edge_types=["corr"]
    )

    arts_empty = GraphNetworkModel(cfg_empty).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    arts_no_smooth = GraphNetworkModel(cfg_no_smooth).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )

    # feature_scores come from the original switch_matrix (pre-smoothing), identical in both
    fs_empty = arts_empty.feature_scores.sort_values("gene_id").reset_index(drop=True)
    fs_no = arts_no_smooth.feature_scores.sort_values("gene_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(fs_empty, fs_no, check_exact=False, rtol=1e-10)

    assert len(arts_empty.edge_table) == len(arts_no_smooth.edge_table), (
        f"empty_graph has {len(arts_empty.edge_table)} edges, "
        f"no_smoothing has {len(arts_no_smooth.edge_table)}"
    )


# ---------------------------------------------------------------------------
# Prior edges enriched within true modules (interpretability gate)
# ---------------------------------------------------------------------------


def test_graph_prior_edges_enriched_in_modules(tmp_path: Path) -> None:
    """Corr-based prior graph edges are enriched within true modules on medium_v1.

    This is the primary Stage-3 promotion gate: the biological prior graph
    captures module structure without being given ground-truth labels.
    """
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    medium_dir = next(p for p in paths if "medium" in p.name)
    bundle = load_dataset_bundle(medium_dir)

    artifacts, _ = _fit_graph(medium_dir, _MEDIUM_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet")
    assert truth is not None and not truth.empty

    from isograph.features.residualize import build_design_matrix, residualize_rows
    from isograph.features.switch import gene_switch_coordinates

    switch_matrix, feature_info = gene_switch_coordinates(
        bundle.matrices["transcript_counts"],
        bundle.feature_tables["transcript"],
    )
    if switch_matrix.size:
        design = build_design_matrix(bundle.sample_table, _MEDIUM_CONFIG.residualize_covariates)
        switch_matrix = residualize_rows(switch_matrix, design)

    gene_ids = feature_info["gene_id"].tolist()
    gene_graph = build_gene_graph(
        switch_matrix, gene_ids, bundle.feature_tables["transcript"],
        _MEDIUM_CONFIG.edge_types, _MEDIUM_CONFIG.corr_threshold,
    )

    report = graph_prior_edge_report(gene_graph, artifacts.edge_table, truth)

    assert report["n_prior_edges"] > 0, "No prior edges were constructed — check corr_threshold"
    assert report["within_module_prior_fraction"] is not None
    assert report["between_module_prior_fraction"] is not None

    within_rate = report.get("within_module_edge_rate", 0.0)
    between_rate = report.get("between_module_edge_rate", 0.0)
    within_weight = report.get("within_module_mean_weight", 0.0)
    between_weight = report.get("between_module_mean_weight", 0.0)

    # When the graph is dense (rates saturate at ~1.0), check mean edge weight instead.
    # Within-module gene pairs always have higher switch-coordinate correlations than
    # between-module pairs, so mean |r| must be higher within modules regardless of density.
    if within_rate >= 0.99 and between_rate >= 0.99:
        assert within_weight > between_weight, (
            f"Prior edges NOT enriched within modules (weight comparison — graph is dense): "
            f"within_mean_weight={within_weight:.4f}, between_mean_weight={between_weight:.4f}"
        )
    else:
        assert within_rate > between_rate, (
            f"Prior edges NOT enriched within modules (rate comparison): "
            f"within_rate={within_rate:.4f}, between_rate={between_rate:.4f}"
        )


# ---------------------------------------------------------------------------
# Graph ablation report structure
# ---------------------------------------------------------------------------


def test_graph_ablation_report_structure(tmp_path: Path) -> None:
    """graph_ablation_report returns the expected structure for all ablation labels."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    medium_dir = next(p for p in paths if "medium" in p.name)
    bundle = load_dataset_bundle(medium_dir)
    truth = bundle.truth_tables.get("truth_modules.parquet")

    ablations = [
        ("full_graph", GraphModelConfig(alpha=0.01, min_module_size=2, gamma=0.5, edge_types=["corr"])),
        ("no_smoothing", GraphModelConfig(alpha=0.01, min_module_size=2, gamma=0.0, edge_types=["corr"])),
        ("empty_graph", GraphModelConfig(alpha=0.01, min_module_size=2, gamma=0.5, edge_types=[])),
    ]

    report = graph_ablation_report(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
        truth_modules=truth,
        ablation_configs=ablations,
    )

    assert "ablations" in report
    assert len(report["ablations"]) == 3

    labels = [r["label"] for r in report["ablations"]]
    assert "full_graph" in labels
    assert "no_smoothing" in labels
    assert "empty_graph" in labels

    for row in report["ablations"]:
        assert "recovery" in row
        assert "n_edges" in row
        assert "n_modules" in row
        assert "calibration" in row
        assert row["recovery"] is not None
        assert isinstance(row["n_edges"], int)

    # no_smoothing and empty_graph should have equal edge counts (both are L=0 effective)
    no_smooth_row = next(r for r in report["ablations"] if r["label"] == "no_smoothing")
    empty_row = next(r for r in report["ablations"] if r["label"] == "empty_graph")
    assert no_smooth_row["n_edges"] == empty_row["n_edges"], (
        "no_smoothing and empty_graph should produce identical edge counts"
    )


# ---------------------------------------------------------------------------
# Benchmark runner integration (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_graph_benchmark_runner(tmp_path: Path) -> None:
    """benchmark() with backend=graph writes expected files and clears recovery gates."""
    config = BenchmarkCommandConfig(
        dataset_root=tmp_path / "datasets",
        artifacts_root=tmp_path / "artifacts",
        report_root=tmp_path / "reports",
        fixture_filter="toy_v1",
        backend="graph",
        stage_name="stage3_graph",
        seed=7,
        graph=GraphModelConfig(alpha=0.05, min_module_size=2, gamma=0.5),
        fixture_graph_overrides={"toy_v1": {"alpha": 0.05, "min_module_size": 2, "gamma": 0.5}},
        recovery_thresholds={"toy_v1": 1.0},
    )
    result = benchmark(config)

    assert result["report"].exists(), "Benchmark report not written"
    assert result["runtime_memory"].exists(), "Runtime/memory report not written"
    assert "calibration" in result, "Calibration report not written"
    assert result["calibration"].exists()

    data = json.loads(result["report"].read_text())
    assert len(data["results"]) == 1
    toy_row = data["results"][0]
    assert toy_row["dataset"] == "toy_v1"
    assert toy_row["recovery"] == 1.0, f"toy_v1 graph recovery={toy_row['recovery']:.4f}, expected 1.0"
    assert data["gate_failures"] == [], f"Unexpected gate failures: {data['gate_failures']}"
    assert data["backend"] == "graph"

    cal_data = json.loads(result["calibration"].read_text())
    assert len(cal_data["calibration_by_fixture"]) == 1
    cal_row = cal_data["calibration_by_fixture"][0]
    assert cal_row["fixture"] == "toy_v1"
    assert "graph_n_edges" in cal_row, "graph_n_edges missing from calibration report"
    assert cal_row["mean_log_likelihood"] is not None


# ---------------------------------------------------------------------------
# Runtime budget (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_graph_runtime_budget_medium_v1(tmp_path: Path) -> None:
    """GraphNetworkModel.fit on medium_v1 completes in under 300 seconds."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    medium_dir = next(p for p in paths if "medium" in p.name)
    bundle = load_dataset_bundle(medium_dir)

    model = GraphNetworkModel(_MEDIUM_CONFIG)
    t0 = time.perf_counter()
    model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 300.0, f"GraphNetworkModel.fit took {elapsed:.1f}s, exceeds 300s budget"
