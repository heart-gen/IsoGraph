"""Stage 1 promotion gate tests.

Covers items from the Stage 1 checklist that are not already in the smoke suite:
  - Sample-order permutation invariance
  - Single-isoform gene handling
  - Zero-count gene edge case
  - medium_v1 recovery threshold (≥ 0.7)
  - medium_v1 snapshot determinism
  - Benchmark runner integration (synthetic fixtures only, marked slow)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.benchmarks.synthetic import generate_core_suite
from isograph.evaluation.metrics import module_recovery_score
from isograph.evaluation.runner import benchmark
from isograph.evaluation.snapshots import compare_snapshot_dirs, save_snapshot
from isograph.io.artifacts import (
    DatasetBundle,
    build_feature_spec,
    build_matrix_spec,
    load_dataset_bundle,
    save_dataset_bundle,
)
from isograph.models.baseline import BaselineNetworkModel
from isograph.validation import DatasetManifest
from isograph.workflow.config import BaselineModelConfig, BenchmarkCommandConfig

_TOY_CONFIG = BaselineModelConfig(alpha=0.05, min_module_size=2)
_MEDIUM_CONFIG = BaselineModelConfig(alpha=0.02, min_module_size=2)


def _fit(dataset_dir: Path, config: BaselineModelConfig):
    from isograph.io.artifacts import load_dataset_bundle

    bundle = load_dataset_bundle(dataset_dir)
    model = BaselineNetworkModel(config)
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
# Permutation invariance
# ---------------------------------------------------------------------------


def test_feature_permutation_invariance(tmp_path: Path) -> None:
    """Network structure is identical regardless of sample order in the input."""
    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    toy_dir = next(p for p in paths if "toy" in p.name)

    bundle = load_dataset_bundle(toy_dir)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(bundle.sample_table))

    model = BaselineNetworkModel(_TOY_CONFIG)

    arts_orig = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    arts_perm = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"][:, perm],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table.iloc[perm].reset_index(drop=True),
    )

    # Module membership must be identical
    orig_modules = (
        arts_orig.module_table.sort_values(["module_id", "gene_id"]).reset_index(drop=True)
    )
    perm_modules = (
        arts_perm.module_table.sort_values(["module_id", "gene_id"]).reset_index(drop=True)
    )
    assert orig_modules.equals(perm_modules), "Module table changed after sample permutation"

    # Edge connectivity must be identical (same gene pairs above threshold).
    # Edge weights may differ in sign (stable_sign convention can flip per sample order),
    # but |weight| and which pairs are included must be invariant.
    def _edge_pairs(et: pd.DataFrame) -> set[frozenset]:
        return {frozenset([row.source, row.target]) for row in et.itertuples()}

    assert _edge_pairs(arts_orig.edge_table) == _edge_pairs(arts_perm.edge_table), (
        "Edge connectivity changed after sample permutation"
    )

    # |weight| values must match within tolerance
    def _abs_weight_map(et: pd.DataFrame) -> dict[frozenset, float]:
        return {frozenset([r.source, r.target]): abs(r.weight) for r in et.itertuples()}

    orig_w = _abs_weight_map(arts_orig.edge_table)
    perm_w = _abs_weight_map(arts_perm.edge_table)
    for pair, w in orig_w.items():
        assert abs(w - perm_w[pair]) < 1e-9, (
            f"Edge weight magnitude changed after permutation for {pair}: {w} vs {perm_w[pair]}"
        )


# ---------------------------------------------------------------------------
# Single-isoform gene handling
# ---------------------------------------------------------------------------


def test_single_isoform_gene_handling(tmp_path: Path) -> None:
    """A gene with only one transcript is excluded from feature_scores without crashing."""
    n_samples = 30
    rng = np.random.default_rng(0)

    # Gene A: 2 transcripts (has switch axis)
    # Gene B: 1 transcript (no switch axis — should be skipped)
    transcript_table = pd.DataFrame(
        {
            "transcript_id": ["A_T1", "A_T2", "B_T1"],
            "gene_id": ["GeneA", "GeneA", "GeneB"],
            "length": [1000, 900, 1000],
        }
    )
    # transcript_counts: shape (3, n_samples)
    base = rng.gamma(5, 10, (3, n_samples))
    transcript_counts = np.floor(base).astype(float)

    bundle = _minimal_bundle(
        n_genes=2,
        n_samples=n_samples,
        transcript_counts=transcript_counts,
        transcript_table=transcript_table,
        gene_ids=["GeneA", "GeneB"],
    )
    dataset_dir = save_dataset_bundle(bundle, tmp_path / "single_isoform")

    config = BaselineModelConfig(alpha=0.05, min_module_size=1)
    model = BaselineNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )

    gene_ids_in_scores = set(artifacts.feature_scores["gene_id"].tolist())
    assert "GeneB" not in gene_ids_in_scores, (
        "Single-isoform GeneB should be excluded from feature_scores"
    )
    assert "GeneA" in gene_ids_in_scores, "GeneA (2 transcripts) should be present"


# ---------------------------------------------------------------------------
# Zero-count gene edge case
# ---------------------------------------------------------------------------


def test_zero_counts_edge_case(tmp_path: Path) -> None:
    """A gene with all-zero transcript counts does not crash the pipeline."""
    n_samples = 30
    rng = np.random.default_rng(1)

    # Gene A: normal counts; Gene B: all zeros
    transcript_table = pd.DataFrame(
        {
            "transcript_id": ["A_T1", "A_T2", "B_T1", "B_T2"],
            "gene_id": ["GeneA", "GeneA", "GeneB", "GeneB"],
            "length": [1000, 900, 1000, 900],
        }
    )
    counts = rng.gamma(5, 10, (4, n_samples))
    counts = np.floor(counts).astype(float)
    counts[2:, :] = 0.0  # zero out GeneB

    bundle = _minimal_bundle(
        n_genes=2,
        n_samples=n_samples,
        transcript_counts=counts,
        transcript_table=transcript_table,
        gene_ids=["GeneA", "GeneB"],
    )

    config = BaselineModelConfig(alpha=0.05, min_module_size=1)
    model = BaselineNetworkModel(config)
    # Must not raise
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    assert isinstance(artifacts.feature_scores, pd.DataFrame)


# ---------------------------------------------------------------------------
# medium_v1 recovery threshold gate (≥ 0.7)
# ---------------------------------------------------------------------------


def test_medium_v1_recovery_threshold(tmp_path: Path) -> None:
    """medium_v1 module recovery clears the Stage 1 threshold of 0.7."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    medium_dir = next(p for p in paths if "medium" in p.name)

    artifacts, bundle = _fit(medium_dir, _MEDIUM_CONFIG)
    truth = bundle.truth_tables.get("truth_modules.parquet")
    assert truth is not None and not truth.empty

    recovery = module_recovery_score(artifacts.module_table, truth)
    assert recovery >= 0.7, f"medium_v1 recovery {recovery:.4f} < Stage 1 gate 0.7"


# ---------------------------------------------------------------------------
# medium_v1 snapshot determinism
# ---------------------------------------------------------------------------


def test_core_v1_determinism_medium_v1(tmp_path: Path) -> None:
    """Two fits on medium_v1 with the same seed produce identical snapshots."""
    from time import perf_counter

    paths = generate_core_suite(tmp_path / "datasets", seed=0)
    medium_dir = next(p for p in paths if "medium" in p.name)

    snap_a = tmp_path / "snap_a"
    snap_b = tmp_path / "snap_b"

    for snap_dir in (snap_a, snap_b):
        artifacts, bundle = _fit(medium_dir, _MEDIUM_CONFIG)
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
            snapshot_name="stage1_medium_v1_baseline_v1_seed0000",
            dataset_name="medium_v1",
        )

    report = compare_snapshot_dirs(snap_a, snap_b)
    assert report["passed"], (
        "medium_v1 snapshots are not deterministic:\n" + "\n".join(report["differences"])
    )


# ---------------------------------------------------------------------------
# Benchmark runner integration (slow — requires no real data)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_benchmark_runner_synthetic_only(tmp_path: Path) -> None:
    """benchmark() with fixture_filter='toy_v1' writes expected files and hits recovery gate."""
    config = BenchmarkCommandConfig(
        dataset_root=tmp_path / "datasets",
        artifacts_root=tmp_path / "artifacts",
        report_root=tmp_path / "reports",
        fixture_filter="toy_v1",
        seed=7,
        model=_TOY_CONFIG,
    )
    result = benchmark(config)

    assert result["report"].exists(), "Benchmark report not written"
    assert result["runtime_memory"].exists(), "Runtime/memory report not written"

    data = json.loads(result["report"].read_text())
    assert len(data["results"]) == 1
    toy_row = data["results"][0]
    assert toy_row["dataset"] == "toy_v1"
    assert toy_row["recovery"] == 1.0, f"toy_v1 recovery={toy_row['recovery']:.4f}, expected 1.0"
    assert data["gate_failures"] == [], f"Unexpected gate failures: {data['gate_failures']}"

    rt_data = json.loads(result["runtime_memory"].read_text())
    assert rt_data["results"][0]["peak_memory_bytes"] > 0
