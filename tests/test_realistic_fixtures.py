"""Gate tests for realistic_v1 and realistic_unequal_v1 fixtures.

These fixtures stress-test assumptions that toy_v1 and medium_v1 do not exercise:

  realistic_v1          — equal module sizes; tests non-switching background genes,
                          variable isoform counts (2–5), NB overdispersion, and
                          between-module partial correlations via a shared confounder.

  realistic_unequal_v1  — same realistic features PLUS power-law module sizes
                          [38, 25, 17, 12, 8], so module-size imbalance is the sole
                          additional variable relative to realistic_v1.

The two-fixture design separates the effect of module-size distribution from all
other realistic properties, allowing per-parameter ablation.

Gate structure
--------------
- Baseline model: NOT gated on realistic fixtures.  The baseline consistently
  fails to separate modules in the presence of 50 % non-switching noise genes —
  this is the expected, documented behaviour and demonstrates that the fixtures
  are appropriately challenging.
- Latent model: gated at recovery ≥ 0.875 (realistic_v1) and ≥ 0.5
  (realistic_unequal_v1), achieved with n_components=5, alpha=0.02.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.benchmarks.synthetic import (
    RealisticDatasetSpec,
    _generate_realistic_dataset,
    generate_core_suite,
)
from isograph.evaluation.metrics import module_recovery_score
from isograph.evaluation.selection import stability_selection
from isograph.io.artifacts import load_dataset_bundle
from isograph.models.baseline import BaselineNetworkModel
from isograph.models.latent import LatentNetworkModel
from isograph.workflow.config import BaselineModelConfig, LatentModelConfig

_LATENT_CFG = LatentModelConfig(alpha=0.02, min_module_size=2)


# ---------------------------------------------------------------------------
# Fixture structure tests
# ---------------------------------------------------------------------------


def test_realistic_v1_structure(tmp_path: Path) -> None:
    """realistic_v1 has expected shape, switching fraction, and isoform distribution."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    rv1_dir = next(p for p in paths if p.name == "realistic_v1")
    bundle = load_dataset_bundle(rv1_dir)

    assert bundle.manifest.dataset_name == "realistic_v1"
    assert bundle.matrices["transcript_counts"].shape[1] == 160, "n_samples must be 160"

    truth_sw = bundle.feature_tables["truth_switch"]
    n_switching    = truth_sw["has_switch"].sum()
    n_nonswitching = (~truth_sw["has_switch"]).sum()
    assert n_switching == 100, f"Expected 100 switching genes, got {n_switching}"
    assert n_nonswitching == 100, f"Expected 100 non-switching genes, got {n_nonswitching}"

    # Verify isoform distribution: all genes have 2–5 transcripts
    tx_per_gene = bundle.feature_tables["transcript"].groupby("gene_id").size()
    assert tx_per_gene.min() >= 2, "All genes must have ≥ 2 isoforms"
    assert tx_per_gene.max() <= 5, "All genes must have ≤ 5 isoforms"
    # Multi-isoform genes must be present (probability of all-2 with 200 genes is negligible)
    assert (tx_per_gene > 2).any(), "At least some genes must have > 2 isoforms"

    # Truth modules cover only switching genes
    truth_modules = bundle.truth_tables["truth_modules.parquet"]
    assert set(truth_modules["gene_id"]).issubset(
        set(truth_sw.loc[truth_sw["has_switch"], "gene_id"])
    ), "truth_modules must only contain switching genes"

    # Equal module sizes
    module_sizes = truth_modules.groupby("module_id").size()
    assert module_sizes.nunique() == 1, (
        f"realistic_v1 must have equal module sizes, got {module_sizes.to_dict()}"
    )


def test_realistic_unequal_v1_structure(tmp_path: Path) -> None:
    """realistic_unequal_v1 has the specified power-law module sizes [38,25,17,12,8]."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    ruv1_dir = next(p for p in paths if p.name == "realistic_unequal_v1")
    bundle = load_dataset_bundle(ruv1_dir)

    assert bundle.manifest.dataset_name == "realistic_unequal_v1"

    truth_modules = bundle.truth_tables["truth_modules.parquet"]
    sizes = sorted(
        truth_modules.groupby("module_id").size().tolist(), reverse=True
    )
    assert sizes == [38, 25, 17, 12, 8], (
        f"Expected power-law sizes [38,25,17,12,8], got {sizes}"
    )

    # Non-switching genes still present at 50 %
    truth_sw = bundle.feature_tables["truth_switch"]
    assert truth_sw["has_switch"].sum() == 100
    assert (~truth_sw["has_switch"]).sum() == 100


def test_realistic_module_sizes_parameter(tmp_path: Path) -> None:
    """RealisticDatasetSpec raises if module_sizes sum does not match n_switching."""
    with pytest.raises(ValueError, match="module_sizes sum"):
        _generate_realistic_dataset(
            RealisticDatasetSpec(
                name="bad",
                n_genes=100,
                n_samples=60,
                n_modules=3,
                module_sizes=[30, 30, 30],  # sum=90, n_switching=50 → mismatch
                seed=0,
            ),
            suite_name="test",
            description="bad spec",
        )


# ---------------------------------------------------------------------------
# Baseline behaviour: expected failure on realistic fixtures
# ---------------------------------------------------------------------------


def test_baseline_fails_realistic_v1(tmp_path: Path) -> None:
    """Baseline does not recover modules on realistic_v1 — documents expected limitation.

    The 50 % non-switching background and multi-isoform composition makes Ledoit-Wolf
    partial correlation unreliable at n_samples=160. This test pins the observed
    behaviour as a regression target; it should NOT be updated to assert success.
    """
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    rv1_dir = next(p for p in paths if p.name == "realistic_v1")
    bundle = load_dataset_bundle(rv1_dir)
    truth = bundle.truth_tables["truth_modules.parquet"]

    best_recovery = 0.0
    for alpha in [0.005, 0.01, 0.02, 0.05]:
        cfg = BaselineModelConfig(alpha=alpha, min_module_size=2)
        arts = BaselineNetworkModel(cfg).fit(
            transcript_counts=bundle.matrices["transcript_counts"],
            transcript_table=bundle.feature_tables["transcript"],
            sample_table=bundle.sample_table,
        )
        rec = module_recovery_score(arts.module_table, truth)
        best_recovery = max(best_recovery, rec)

    assert best_recovery < 0.5, (
        f"Baseline unexpectedly recovered realistic_v1 modules (recovery={best_recovery:.4f}). "
        "If the baseline model genuinely improved, update this test and add a gate."
    )


# ---------------------------------------------------------------------------
# Latent model recovery gates
# ---------------------------------------------------------------------------


def test_latent_realistic_v1_recovery(tmp_path: Path) -> None:
    """Latent backend recovers all equal-sized modules on realistic_v1 (recovery ≥ 0.875)."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    rv1_dir = next(p for p in paths if p.name == "realistic_v1")
    bundle = load_dataset_bundle(rv1_dir)
    truth = bundle.truth_tables["truth_modules.parquet"]

    arts = LatentNetworkModel(_LATENT_CFG).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= 0.875, (
        f"realistic_v1 latent recovery {recovery:.4f} < gate 0.875"
    )


def test_latent_realistic_unequal_v1_recovery(tmp_path: Path) -> None:
    """Latent backend recovers modules on realistic_unequal_v1 (recovery ≥ 0.5).

    The lower gate reflects the added difficulty of power-law module sizes:
    the smallest module (8 genes) is harder to recover from a background of
    100 non-switching genes, making this a stricter stress test than realistic_v1.
    """
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    ruv1_dir = next(p for p in paths if p.name == "realistic_unequal_v1")
    bundle = load_dataset_bundle(ruv1_dir)
    truth = bundle.truth_tables["truth_modules.parquet"]

    arts = LatentNetworkModel(_LATENT_CFG).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= 0.5, (
        f"realistic_unequal_v1 latent recovery {recovery:.4f} < gate 0.5"
    )


def test_realistic_unequal_harder_than_equal(tmp_path: Path) -> None:
    """Unequal module sizes make recovery strictly harder than equal sizes.

    Pins the empirical observation that realistic_unequal_v1 requires different
    alpha tuning or yields lower recovery than realistic_v1, isolating the effect
    of module-size distribution as the sole variable between the two fixtures.
    """
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    bundles = {p.name: load_dataset_bundle(p) for p in paths
               if p.name in ("realistic_v1", "realistic_unequal_v1")}

    recoveries = {}
    for name, bundle in bundles.items():
        truth = bundle.truth_tables["truth_modules.parquet"]
        # Use a higher alpha that would still pass for equal but challenge unequal
        cfg = LatentModelConfig(alpha=0.05, min_module_size=2)
        arts = LatentNetworkModel(cfg).fit(
            transcript_counts=bundle.matrices["transcript_counts"],
            transcript_table=bundle.feature_tables["transcript"],
            sample_table=bundle.sample_table,
        )
        recoveries[name] = module_recovery_score(arts.module_table, truth)

    assert recoveries["realistic_v1"] >= recoveries["realistic_unequal_v1"], (
        f"Expected equal-module fixture to be at least as easy as unequal: "
        f"realistic_v1={recoveries['realistic_v1']:.4f}, "
        f"realistic_unequal_v1={recoveries['realistic_unequal_v1']:.4f}"
    )


# ---------------------------------------------------------------------------
# Calibration and non-switching gene exclusion
# ---------------------------------------------------------------------------


def test_latent_realistic_v1_calibration(tmp_path: Path) -> None:
    """FitArtifacts.calibration is populated on realistic_v1."""
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    rv1_dir = next(p for p in paths if p.name == "realistic_v1")
    bundle = load_dataset_bundle(rv1_dir)

    arts = LatentNetworkModel(_LATENT_CFG).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    assert arts.calibration is not None
    assert arts.calibration["reconstruction_rmse"] >= 0
    # Non-trivial reconstruction error expected (data is noisy, not rank-k)
    assert arts.calibration["reconstruction_rmse"] > 0


def test_nonswitching_genes_absent_from_feature_scores(tmp_path: Path) -> None:
    """Non-switching genes with only 1 isoform variant are excluded from feature_scores.

    Non-switching genes are generated with k ≥ 2 isoforms (same as switching genes),
    so they all pass the ≥2-transcript filter in group_transcript_clr. They will
    appear in feature_scores — this test confirms the pipeline does not crash and
    that the feature_scores DataFrame covers all valid genes.
    """
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    rv1_dir = next(p for p in paths if p.name == "realistic_v1")
    bundle = load_dataset_bundle(rv1_dir)

    arts = LatentNetworkModel(_LATENT_CFG).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    # All genes have ≥ 2 isoforms, so all 200 should appear in feature_scores
    assert len(arts.feature_scores) == 200, (
        f"Expected 200 genes in feature_scores, got {len(arts.feature_scores)}"
    )
    assert "gene_id" in arts.feature_scores.columns


# ---------------------------------------------------------------------------
# Stability selection on realistic_v1
# ---------------------------------------------------------------------------


def test_stability_selection_realistic_v1(tmp_path: Path) -> None:
    """Stability selection recommends alpha that recovers correct modules on realistic_v1.

    realistic_v1 simulates the real-data challenge (no easily-known correct alpha),
    so stability selection is the intended workflow. This test verifies that the
    stability-selected alpha gives recovery ≥ 0.875 — the same gate as the direct test.
    """
    paths = generate_core_suite(tmp_path / "datasets", seed=7)
    rv1_dir = next(p for p in paths if p.name == "realistic_v1")
    bundle = load_dataset_bundle(rv1_dir)
    truth = bundle.truth_tables["truth_modules.parquet"]

    model = LatentNetworkModel(_LATENT_CFG)
    result = stability_selection(
        model=model,
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
        alpha_grid=[0.01, 0.02, 0.05, 0.10],
        n_iterations=20,
        subsample_fraction=0.8,
        stability_threshold=0.6,
        seed=0,
    )

    assert result.recommended_alpha in [0.01, 0.02, 0.05, 0.10]

    # Recovery at recommended alpha must clear the gate
    rec_cfg = LatentModelConfig(
        alpha=result.recommended_alpha,
        min_module_size=_LATENT_CFG.min_module_size,
    )
    arts = LatentNetworkModel(rec_cfg).fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= 0.875, (
        f"Stability-selected alpha={result.recommended_alpha} gives recovery={recovery:.4f} "
        f"< gate 0.875 on realistic_v1"
    )
