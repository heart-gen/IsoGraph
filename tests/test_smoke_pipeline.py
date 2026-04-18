"""End-to-end smoke tests for the benchmark pipeline on synthetic and real fixtures.

Synthetic tests use lower alpha thresholds than the default BaselineModelConfig
because synthetic partial correlations are weaker than real-data ones (the
default alpha=0.12 is calibrated for real BrainSeq data with ~256 genes and
~300 samples).  The structural correctness and determinism checks here are
independent of that tuning.

Real-data tests are skipped automatically when ``data/counts/gene-counts.tsv.gz``
is absent (i.e. on CI runners without access to the BrainSeq files).
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pytest

from isograph.benchmarks.synthetic import generate_core_suite
from isograph.evaluation.metrics import module_recovery_score
from isograph.evaluation.snapshots import compare_snapshot_dirs, save_snapshot
from isograph.io.artifacts import load_dataset_bundle
from isograph.models.baseline import BaselineNetworkModel
from isograph.workflow.config import BaselineModelConfig, BenchmarkCommandConfig, RealDataFreezeConfig
from isograph.io.real_data import freeze_real_dataset
from isograph.utils import ensure_dir

# Real data is available when this file exists locally.
_REAL_DATA_AVAILABLE = Path("data/counts/gene-counts.tsv.gz").exists()
_REAL_FIXTURE_DIR = Path("benchmarks/datasets/core_v1/real_caudate_aa_v1")
_REAL_SNAPSHOT_DIR = Path("snapshots/stage0_real_caudate_aa_v1_baseline_v1_seed0000")

requires_real_data = pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="BrainSeq data not available (data/counts/gene-counts.tsv.gz missing)",
)

# Config tuned for toy_v1 (24 genes, 48 samples): alpha=0.05 reliably finds
# the two planted modules with perfect recovery.
_TOY_CONFIG = BaselineModelConfig(alpha=0.05, min_module_size=2)

# Config tuned for medium_v1 (400 genes, 240 samples): alpha=0.03 yields ~8–12
# modules, consistent with the 8 planted modules in the spec.
_MEDIUM_CONFIG = BaselineModelConfig(alpha=0.03, min_module_size=2)


def _fit_bundle(dataset_dir: Path, config: BaselineModelConfig) -> tuple:
    """Load a bundle, fit the model, and return (artifacts, bundle, elapsed)."""
    bundle = load_dataset_bundle(dataset_dir)
    model = BaselineNetworkModel(config)
    start = perf_counter()
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    elapsed = perf_counter() - start
    return artifacts, bundle, elapsed


def test_smoke_toy_v1(tmp_path: Path) -> None:
    """Pipeline runs end-to-end on toy_v1 and produces all four artifact tables."""
    import pandas as pd

    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    toy_dir = next(p for p in paths if "toy" in p.name)

    artifacts, bundle, _ = _fit_bundle(toy_dir, _TOY_CONFIG)

    assert isinstance(artifacts.module_table, pd.DataFrame)
    assert isinstance(artifacts.edge_table, pd.DataFrame)
    assert isinstance(artifacts.trait_table, pd.DataFrame)
    assert isinstance(artifacts.feature_scores, pd.DataFrame)

    # feature_scores has one row per gene
    n_genes = bundle.feature_tables["transcript"]["gene_id"].nunique()
    assert len(artifacts.feature_scores) == n_genes
    assert "gene_id" in artifacts.feature_scores.columns


def test_smoke_medium_v1(tmp_path: Path) -> None:
    """Pipeline runs on medium_v1 and discovers at least one module."""
    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    medium_dir = next(p for p in paths if "medium" in p.name)

    artifacts, _, _ = _fit_bundle(medium_dir, _MEDIUM_CONFIG)

    assert not artifacts.module_table.empty, "Expected at least one module on medium_v1"
    assert artifacts.module_table["module_id"].nunique() > 0


def test_smoke_toy_v1_recovery(tmp_path: Path) -> None:
    """Module recovery on toy_v1 exceeds 0.5 (planted modules are detectable)."""
    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    toy_dir = next(p for p in paths if "toy" in p.name)

    artifacts, bundle, _ = _fit_bundle(toy_dir, _TOY_CONFIG)

    truth = bundle.truth_tables.get("truth_modules.parquet")
    assert truth is not None and not truth.empty, "toy_v1 must carry truth_modules"
    recovery = module_recovery_score(artifacts.module_table, truth)
    assert recovery > 0.5, f"Expected recovery > 0.5, got {recovery:.4f}"


def test_snapshot_deterministic(tmp_path: Path) -> None:
    """Two fits with the same seed produce identical snapshots."""
    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    toy_dir = next(p for p in paths if "toy" in p.name)

    snap_a = tmp_path / "snap_a"
    snap_b = tmp_path / "snap_b"

    for snap_dir in (snap_a, snap_b):
        artifacts, bundle, elapsed = _fit_bundle(toy_dir, _TOY_CONFIG)
        truth = bundle.truth_tables.get("truth_modules.parquet")
        recovery = module_recovery_score(artifacts.module_table, truth) if truth is not None else None
        metrics = {
            "n_modules": (
                0 if artifacts.module_table.empty
                else artifacts.module_table["module_id"].nunique()
            ),
            "n_edges": len(artifacts.edge_table),
            "recovery": recovery,
            "runtime_seconds": elapsed,
        }
        save_snapshot(
            fit_artifacts=artifacts,
            model_config=_TOY_CONFIG,
            metrics=metrics,
            output_dir=snap_dir,
            snapshot_name="stage0_toy_v1_baseline_v1_seed0000",
            dataset_name="toy_v1",
        )

    report = compare_snapshot_dirs(snap_a, snap_b)
    assert report["passed"], (
        "Snapshots are not deterministic:\n" + "\n".join(report["differences"])
    )


# ---------------------------------------------------------------------------
# Real-data smoke tests (skipped on CI; require data/counts/gene-counts.tsv.gz)
# ---------------------------------------------------------------------------

@requires_real_data
def test_smoke_real_caudate_aa_v1_freeze(tmp_path: Path) -> None:
    """freeze_real_dataset produces a valid bundle for real_caudate_aa_v1."""
    config = BenchmarkCommandConfig()
    suite_dir = ensure_dir(tmp_path / "datasets" / "core_v1")
    dataset_dir = freeze_real_dataset(config.real_data, suite_dir)
    bundle = load_dataset_bundle(dataset_dir)

    assert sorted(bundle.feature_tables) == ["gene", "psi", "transcript"]
    assert bundle.matrices["transcript_counts"].shape[1] == len(bundle.sample_table)
    assert bundle.matrices["gene_counts"].shape[0] == len(bundle.feature_tables["gene"])
    assert bundle.manifest.provenance.get("splicing_feature_type") == "psi"


@requires_real_data
def test_smoke_real_caudate_aa_v1_fit() -> None:
    """Baseline model fits on the frozen real_caudate_aa_v1 fixture."""
    assert _REAL_FIXTURE_DIR.exists(), (
        f"Run 'isograph freeze-real' first to build {_REAL_FIXTURE_DIR}"
    )
    bundle = load_dataset_bundle(_REAL_FIXTURE_DIR)
    config = BaselineModelConfig()
    artifacts = BaselineNetworkModel(config).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )

    assert "gene_id" in artifacts.feature_scores.columns
    # Genes with only 1 transcript are dropped by gene_switch_coordinates (no switch axis).
    # feature_scores may have fewer rows than unique genes in the transcript table.
    n_panel_genes = len(bundle.feature_tables["transcript"]["gene_id"].unique())
    assert 0 < len(artifacts.feature_scores) <= n_panel_genes
    assert isinstance(artifacts.module_table.shape, tuple)


@requires_real_data
def test_snapshot_real_caudate_aa_v1_matches_reference(tmp_path: Path) -> None:
    """Re-fit on real_caudate_aa_v1 and compare against committed reference snapshot."""
    assert _REAL_FIXTURE_DIR.exists(), (
        f"Run 'isograph freeze-real' first to build {_REAL_FIXTURE_DIR}"
    )
    assert _REAL_SNAPSHOT_DIR.exists(), (
        f"Reference snapshot missing: {_REAL_SNAPSHOT_DIR}"
    )
    bundle = load_dataset_bundle(_REAL_FIXTURE_DIR)
    config = BaselineModelConfig()
    start = perf_counter()
    artifacts = BaselineNetworkModel(config).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    elapsed = perf_counter() - start
    metrics = {
        "n_modules": (
            0 if artifacts.module_table.empty
            else artifacts.module_table["module_id"].nunique()
        ),
        "n_edges": len(artifacts.edge_table),
        "recovery": None,
        "runtime_seconds": elapsed,
    }
    candidate_dir = tmp_path / "candidate"
    save_snapshot(
        fit_artifacts=artifacts,
        model_config=config,
        metrics=metrics,
        output_dir=candidate_dir,
        snapshot_name="stage0_real_caudate_aa_v1_baseline_v1_seed0000",
        dataset_name="real_caudate_aa_v1",
    )
    report = compare_snapshot_dirs(_REAL_SNAPSHOT_DIR, candidate_dir)
    assert report["passed"], (
        "Real-data snapshot does not match reference:\n" + "\n".join(report["differences"])
    )
