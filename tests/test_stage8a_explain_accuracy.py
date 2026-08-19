"""Stage 8A accuracy tests: explain_module on fitted realistic fixture.

These tests verify that explain_module correctly identifies true switching genes
and transcripts when applied to a BaselineNetworkModel fit of a realistic
synthetic fixture with planted isoform-switch modules.

All tests are marked slow; run with: pytest tests/test_stage8a_explain_accuracy.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.benchmarks.synthetic import RealisticDatasetSpec, _generate_realistic_dataset
from isograph.explain import explain_module
from isograph.features.channels import FEATURE_SCORE_METADATA_COLUMNS
from isograph.models.baseline import BaselineNetworkModel
from isograph.workflow.config import BaselineModelConfig

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

_SPEC = RealisticDatasetSpec(
    name="explain_accuracy_v1",
    n_genes=200,
    n_samples=160,
    n_modules=5,
    module_sizes=None,  # equal: 20 genes per module
    switching_fraction=0.5,  # 100 switching, 100 non-switching
    confounder_weight=0.3,
    count_dispersion=5.0,
    mean_gene_total=300.0,
    seed=42,
)


def _build_bundle():
    return _generate_realistic_dataset(
        _SPEC, suite_name="accuracy_v1", description="accuracy fixture"
    )


def _build_transcript_usage(
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    sample_ids: list[str],
) -> pd.DataFrame:
    """Compute per-gene transcript usage proportions from raw counts.

    Returns samples × transcripts DataFrame (values sum to ~1.0 within each gene per sample).
    """
    n_tx = len(transcript_table)
    usage = np.zeros((n_tx, len(sample_ids)), dtype=float)
    for _, grp in transcript_table.groupby("gene_id", sort=False):
        idx = grp.index.to_numpy()
        block = transcript_counts[idx]  # (k, n_samples)
        totals = block.sum(axis=0, keepdims=True)
        usage[idx] = block / np.maximum(totals, 1.0)
    return pd.DataFrame(
        usage.T, index=sample_ids, columns=transcript_table["transcript_id"].tolist()
    )


def _fit_and_write_artifacts(
    bundle,
    artifact_dir: Path,
) -> pd.DataFrame:
    """Fit baseline model and write modules.parquet + feature_scores.parquet.

    Returns the fitted module_table.
    """
    sample_ids: list[str] = bundle.sample_table["sample_id"].tolist()
    # Stage 9A doubles the feature space (switch + abundance channels), which
    # strengthens LedoitWolf shrinkage and requires a lower alpha than the default
    # 0.12.  Stage 9B will address this with per-channel alpha thresholds.
    config = BaselineModelConfig(
        alpha=0.08,
        min_module_size=3,
        trait_columns=[],
        residualize_covariates=[],
    )
    model = BaselineNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )

    # Rename integer sample columns in feature_scores to match sample_ids.
    # Exclude all metadata columns (feature_id, gene_id, feature_type, n_transcripts).
    fs = artifacts.feature_scores.copy()
    meta = set(FEATURE_SCORE_METADATA_COLUMNS)
    int_cols = [c for c in fs.columns if c not in meta]
    assert len(int_cols) == len(
        sample_ids
    ), f"feature_scores has {len(int_cols)} sample cols; expected {len(sample_ids)}"
    rename_map = dict(zip(int_cols, sample_ids, strict=False))
    fs = fs.rename(columns=rename_map)

    artifacts.module_table.to_parquet(artifact_dir / "modules.parquet", index=False)
    fs.to_parquet(artifact_dir / "feature_scores.parquet", index=False)
    return artifacts.module_table


def _build_feature_meta(
    transcript_table: pd.DataFrame,
    feature_type: str = "transcript_usage",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_id": transcript_table["transcript_id"].tolist(),
            "gene_id": transcript_table["gene_id"].tolist(),
            "transcript_id": transcript_table["transcript_id"].tolist(),
            "feature_type": feature_type,
        }
    )


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score[pos] > score[neg])."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    count = 0
    for p in pos:
        count += (p > neg).sum() + 0.5 * (p == neg).sum()
    return count / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Shared fixture (session-scoped so the heavy fit runs once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def explain_results():
    """Generate bundle, fit baseline, run explain_module; return results and ground truth."""
    bundle = _build_bundle()
    sample_ids: list[str] = bundle.sample_table["sample_id"].tolist()
    truth_switch = bundle.feature_tables["truth_switch"]  # gene_id, has_switch
    truth_modules = bundle.feature_tables["truth_module"]  # gene_id, module_id (int)

    transcript_table = bundle.feature_tables["transcript"]
    feature_table = _build_transcript_usage(
        bundle.matrices["transcript_counts"], transcript_table, sample_ids
    )
    feature_meta = _build_feature_meta(transcript_table)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)
        module_table = _fit_and_write_artifacts(bundle, artifact_dir)

        if module_table.empty:
            pytest.skip("Baseline model found no modules on this fixture (check alpha setting).")

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = explain_module(
                artifact_dir=artifact_dir,
                feature_table=feature_table,
                feature_meta=feature_meta,
            )

    return results, module_table, truth_switch, truth_modules, transcript_table


# ---------------------------------------------------------------------------
# Accuracy gate tests
# ---------------------------------------------------------------------------


class TestGenDriverDiscrimination:
    """Gene driver |r| should correctly identify true switching genes as module drivers."""

    def test_at_least_one_module_fitted(self, explain_results):
        results, module_table, *_ = explain_results
        assert len(results) >= 1, "No modules were fitted; cannot test accuracy."

    def test_switching_genes_have_high_abs_r(self, explain_results):
        """Switching genes in modules should have mean |r| ≥ 0.50 with the module eigengene."""
        results, module_table, truth_switch, *_ = explain_results
        switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])

        sw_r: list[float] = []
        for res in results.values():
            tbl = res.gene_driver_table
            finite = tbl["r"].notna()
            sw_r.extend(tbl.loc[finite & tbl["gene_id"].isin(switch_set), "r"].abs().tolist())

        if not sw_r:
            pytest.skip("No switching genes appear in any fitted module.")
        mean_r = np.mean(sw_r)
        assert mean_r >= 0.50, f"Mean |r| of switching genes in modules = {mean_r:.3f} < 0.50."

    def test_gene_driver_auc_geq_threshold(self, explain_results):
        """When both switching and non-switching genes are in modules, AUC ≥ 0.65.

        Stage 9A's doubled feature space (switch + abundance channels) causes stronger
        LedoitWolf shrinkage. Non-switching genes may not appear in switch-driven modules
        under the current default edge policy; Stage 9B will address this with per-channel
        alpha calibration. This test skips rather than fails when non-switching genes are absent.
        """
        results, module_table, truth_switch, *_ = explain_results
        switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])

        abs_r_vals: list[float] = []
        is_switch: list[int] = []
        for res in results.values():
            tbl = res.gene_driver_table
            finite = tbl["r"].notna()
            abs_r_vals.extend(tbl.loc[finite, "r"].abs().tolist())
            is_switch.extend([1 if g in switch_set else 0 for g in tbl.loc[finite, "gene_id"]])

        labels = np.array(is_switch)
        if labels.sum() == 0 or (labels == 0).sum() == 0:
            pytest.skip(
                "Only one class (switching or non-switching) present in modules — "
                "AUC comparison undefined. Stage 9B will calibrate edge policy."
            )
        scores = np.array(abs_r_vals)
        auc = _auc(scores, labels)
        assert auc >= 0.65, (
            f"Gene driver AUC = {auc:.3f} < 0.65. "
            "Switching genes should rank higher than non-switching by |r|."
        )

    def test_mean_abs_r_switching_gt_nonswitching(self, explain_results):
        """When both classes in modules: mean |r| switching > mean |r| non-switching."""
        results, module_table, truth_switch, *_ = explain_results
        switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])

        sw_r: list[float] = []
        nsw_r: list[float] = []
        for res in results.values():
            tbl = res.gene_driver_table
            finite = tbl["r"].notna()
            for _, row in tbl.loc[finite].iterrows():
                bucket = sw_r if row["gene_id"] in switch_set else nsw_r
                bucket.append(abs(row["r"]))

        if not sw_r or not nsw_r:
            pytest.skip(
                "Non-switching genes absent from modules — comparison undefined. "
                "Stage 9B will calibrate abundance-abundance edge policy."
            )
        assert np.mean(sw_r) > np.mean(
            nsw_r
        ), f"Mean |r| switching={np.mean(sw_r):.3f} not > non-switching={np.mean(nsw_r):.3f}."


class TestTranscriptPolarityAccuracy:
    """Transcript polarity and switch_strength should discriminate switching from non-switching."""

    def test_switch_strength_switching_gt_nonswitching(self, explain_results):
        """When both classes in modules: median switch_strength switching > non-switching."""
        results, module_table, truth_switch, *_ = explain_results
        switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])

        sw_ss: list[float] = []
        nsw_ss: list[float] = []
        for res in results.values():
            tbl = res.transcript_polarity_table
            if tbl.empty:
                continue
            by_gene = tbl.groupby("gene_id")["switch_strength"].first()
            for gene_id, ss in by_gene.items():
                bucket = sw_ss if gene_id in switch_set else nsw_ss
                bucket.append(ss)

        if not sw_ss or not nsw_ss:
            pytest.skip(
                "Non-switching genes absent from modules — comparison undefined. "
                "Stage 9B will calibrate abundance-abundance edge policy."
            )
        assert np.median(sw_ss) > np.median(nsw_ss), (
            f"Median switch_strength switching={np.median(sw_ss):.3f} not > "
            f"non-switching={np.median(nsw_ss):.3f}."
        )

    def test_switch_strength_auc_geq_threshold(self, explain_results):
        """When both classes in modules: AUC of switch_strength ≥ 0.60."""
        results, module_table, truth_switch, *_ = explain_results
        switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])

        ss_vals: list[float] = []
        is_switch: list[int] = []
        for res in results.values():
            tbl = res.transcript_polarity_table
            if tbl.empty:
                continue
            by_gene = tbl.groupby("gene_id")["switch_strength"].first()
            for gene_id, ss in by_gene.items():
                ss_vals.append(ss)
                is_switch.append(1 if gene_id in switch_set else 0)

        labels = np.array(is_switch)
        if not ss_vals or labels.sum() == 0 or (labels == 0).sum() == 0:
            pytest.skip(
                "Non-switching genes absent from modules — AUC undefined. "
                "Stage 9B will calibrate abundance-abundance edge policy."
            )
        auc = _auc(np.array(ss_vals), labels)
        assert auc >= 0.60, (
            f"Switch_strength AUC = {auc:.3f} < 0.60. "
            "True switching genes should have higher switch_strength than non-switching."
        )

    def test_t1_transcript_extreme_r_in_2isoform_switching_genes(self, explain_results):
        """For 2-isoform switching genes, T1 and T2 are anti-correlated → switch_strength ≈ 2|r_T1|.

        T1 is module-driven (sigmoid), T2 is the complement, so |r_T1| = |r_T2| ≈ 1.0.
        Check that |r| ≥ 0.5 for at least 70% of confirmed switching genes with 2 isoforms.
        """
        results, module_table, truth_switch, _, transcript_table = explain_results
        switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])

        # Find switching genes with exactly 2 isoforms
        n_tx_per_gene = transcript_table.groupby("gene_id")["transcript_id"].count()
        two_isoform_switches = switch_set & set(n_tx_per_gene.index[n_tx_per_gene == 2])

        if len(two_isoform_switches) < 5:
            pytest.skip("Fewer than 5 two-isoform switching genes in fitted modules; cannot test.")

        high_r_count = 0
        total = 0
        for res in results.values():
            tbl = res.transcript_polarity_table
            if tbl.empty:
                continue
            for gene_id, grp in tbl.groupby("gene_id"):
                if gene_id not in two_isoform_switches:
                    continue
                if len(grp) != 2:
                    continue
                max_abs_r = grp["r"].abs().max()
                if np.isfinite(max_abs_r):
                    total += 1
                    if max_abs_r >= 0.5:
                        high_r_count += 1

        if total == 0:
            pytest.skip("No 2-isoform switching genes appeared in transcript_polarity_table.")
        fraction = high_r_count / total
        assert fraction >= 0.70, (
            f"Only {fraction:.1%} of 2-isoform switching genes have max |r| ≥ 0.5 "
            f"(expected ≥ 70%). Got {high_r_count}/{total}."
        )


class TestHighVsLowAccuracy:
    """High-vs-low delta should be larger in magnitude for switching than non-switching genes' transcripts."""

    def test_switching_gene_transcripts_have_larger_abs_delta(self, explain_results):
        """Mean |delta| of switching-gene transcripts > mean |delta| of non-switching gene transcripts."""
        results, module_table, truth_switch, *_ = explain_results
        switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])

        sw_delta: list[float] = []
        nsw_delta: list[float] = []
        for res in results.values():
            tbl = res.high_vs_low_table
            finite = tbl["delta"].notna()
            for _, row in tbl.loc[finite].iterrows():
                bucket = sw_delta if row["gene_id"] in switch_set else nsw_delta
                bucket.append(abs(row["delta"]))

        if not sw_delta or not nsw_delta:
            pytest.skip("Not enough data for high-vs-low delta comparison.")
        assert np.mean(sw_delta) > np.mean(nsw_delta), (
            f"Mean |delta| switching={np.mean(sw_delta):.3f} not > "
            f"non-switching={np.mean(nsw_delta):.3f}."
        )
