"""Stage 8A explain accuracy across fixture types (Stage 8A).

Tests whether explain_module correctly identifies switching genes and transcripts
across four structural challenge categories:

  noisy_v1       — high NB dispersion (15×), strong confounder (0.6), small modules
  nonlinear_v1   — radial/product nonlinear module structure (inner ring, outer ring,
                   f1·f2 product), bimodal latent activation
  xlarge_mini    — xlarge_v1 structural parameters (12 modules, ~15% switching
                   fraction, dispersion=7, confounder=0.4) at 1/10 gene scale
  xxlarge_mini   — xxlarge_v1 structural parameters (16 modules, 25% switching
                   fraction, dispersion=7, confounder=0.4) at 1/10 gene scale

All tests are marked slow. Run with:
  pytest tests/test_stage8a_explain_accuracy_suite.py -v
"""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.benchmarks.synthetic import (
    NonlinearDatasetSpec,
    RealisticDatasetSpec,
    _generate_nonlinear_dataset,
    _generate_realistic_dataset,
)
from isograph.explain import explain_module
from isograph.models.baseline import BaselineNetworkModel
from isograph.workflow.config import BaselineModelConfig

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixture specs
# ---------------------------------------------------------------------------

# noisy_v1: exact core_v1 parameters
_NOISY_SPEC = RealisticDatasetSpec(
    name="noisy_explain",
    n_genes=300,
    n_samples=100,
    n_modules=8,
    module_sizes=[22, 18, 14, 11, 9, 8, 7, 6],   # power-law, sum=95
    switching_fraction=95 / 300,
    confounder_weight=0.6,
    count_dispersion=15.0,
    mean_gene_total=300.0,
    dx_effect_range=(0.2, 0.6),
    age_effect_range=(0.1, 0.25),
    switching_concentration=10.0,
    nonswitching_concentration=80.0,
    seed=42,
)

# nonlinear_v1: exact core_v1 parameters
_NONLINEAR_SPEC = NonlinearDatasetSpec(
    name="nonlinear_explain",
    n_genes=200,
    n_samples=200,
    n_modules=4,
    n_nonlinear_modules=4,
    module_sizes=[30, 25, 25, 20],
    switching_fraction=0.5,
    count_dispersion=5.0,
    mean_gene_total=300.0,
    state_effect_size=2.5,
    state_background=0.15,
    confounder_weight=0.2,
    switching_concentration=20.0,
    nonswitching_concentration=100.0,
    seed=42,
)

# xlarge_mini: xlarge_v1 structural params at 1/10 gene scale
# xlarge_v1: 6000 genes, 12 modules, ~15% switching, dispersion=7, confounder=0.4
# Mini: 600 genes, 12 modules equal sizes of 7 = 84 switching (14%)
_XLARGE_MINI_SPEC = RealisticDatasetSpec(
    name="xlarge_mini_explain",
    n_genes=600,
    n_samples=240,
    n_modules=12,
    module_sizes=None,             # equal: 84//12 = 7 genes per module
    switching_fraction=84 / 600,
    confounder_weight=0.4,
    count_dispersion=7.0,
    mean_gene_total=300.0,
    dx_effect_range=(0.4, 0.9),
    age_effect_range=(0.15, 0.45),
    switching_concentration=20.0,
    nonswitching_concentration=100.0,
    seed=42,
)

# xxlarge_mini: xxlarge_v1 structural params at 1/10 gene scale
# xxlarge_v1: 12000 genes, 16 modules, 25% switching, dispersion=7, confounder=0.4
# Mini: 400 genes, 16 modules equal sizes of 6 = 96 switching (24%)
_XXLARGE_MINI_SPEC = RealisticDatasetSpec(
    name="xxlarge_mini_explain",
    n_genes=400,
    n_samples=240,
    n_modules=16,
    module_sizes=None,             # equal: 96//16 = 6 genes per module
    switching_fraction=96 / 400,
    confounder_weight=0.4,
    count_dispersion=7.0,
    mean_gene_total=300.0,
    dx_effect_range=(0.4, 0.9),
    age_effect_range=(0.15, 0.45),
    switching_concentration=20.0,
    nonswitching_concentration=100.0,
    seed=42,
)


# ---------------------------------------------------------------------------
# Shared helpers (identical to test_stage8a_explain_accuracy.py)
# ---------------------------------------------------------------------------

def _make_bundle(spec):
    if isinstance(spec, NonlinearDatasetSpec):
        return _generate_nonlinear_dataset(spec, "accuracy_suite", spec.name)
    return _generate_realistic_dataset(spec, "accuracy_suite", spec.name)


def _build_transcript_usage(
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    sample_ids: list[str],
) -> pd.DataFrame:
    n_tx = len(transcript_table)
    usage = np.zeros((n_tx, len(sample_ids)), dtype=float)
    for _, grp in transcript_table.groupby("gene_id", sort=False):
        idx = grp.index.to_numpy()
        block = transcript_counts[idx]
        totals = block.sum(axis=0, keepdims=True)
        usage[idx] = block / np.maximum(totals, 1.0)
    return pd.DataFrame(usage.T, index=sample_ids, columns=transcript_table["transcript_id"].tolist())


def _fit_and_write_artifacts(bundle, artifact_dir: Path) -> pd.DataFrame:
    sample_ids: list[str] = bundle.sample_table["sample_id"].tolist()
    config = BaselineModelConfig(min_module_size=3, trait_columns=[], residualize_covariates=[])
    model = BaselineNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
    )
    fs = artifacts.feature_scores.copy()
    int_cols = [c for c in fs.columns if c != "gene_id"]
    fs = fs.rename(columns={old: new for old, new in zip(int_cols, sample_ids)})
    artifacts.module_table.to_parquet(artifact_dir / "modules.parquet", index=False)
    fs.to_parquet(artifact_dir / "feature_scores.parquet", index=False)
    return artifacts.module_table


def _run_explain(bundle, artifact_dir: Path):
    sample_ids: list[str] = bundle.sample_table["sample_id"].tolist()
    transcript_table = bundle.feature_tables["transcript"]
    feature_table = _build_transcript_usage(
        bundle.matrices["transcript_counts"], transcript_table, sample_ids
    )
    feature_meta = pd.DataFrame({
        "feature_id": transcript_table["transcript_id"].tolist(),
        "gene_id": transcript_table["gene_id"].tolist(),
        "transcript_id": transcript_table["transcript_id"].tolist(),
        "feature_type": "transcript_usage",
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return explain_module(
            artifact_dir=artifact_dir,
            feature_table=feature_table,
            feature_meta=feature_meta,
        )


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(count / (len(pos) * len(neg)))


@dataclass
class _AccuracyMetrics:
    gene_driver_auc: float
    switch_strength_auc: float
    mean_r_switching: float
    mean_r_nonswitching: float
    median_ss_switching: float
    median_ss_nonswitching: float
    n_fitted_modules: int
    n_genes_in_modules: int


def _compute_accuracy(results, truth_switch) -> _AccuracyMetrics:
    switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])

    abs_r_vals, is_sw_r = [], []
    ss_vals, is_sw_ss = [], []
    for res in results.values():
        tbl = res.gene_driver_table
        finite = tbl["r"].notna()
        abs_r_vals.extend(tbl.loc[finite, "r"].abs().tolist())
        is_sw_r.extend([1 if g in switch_set else 0 for g in tbl.loc[finite, "gene_id"]])

        pt = res.transcript_polarity_table
        if not pt.empty:
            by_gene = pt.groupby("gene_id")["switch_strength"].first()
            for gene_id, ss in by_gene.items():
                ss_vals.append(ss)
                is_sw_ss.append(1 if gene_id in switch_set else 0)

    r_arr = np.array(abs_r_vals)
    sw_arr = np.array(is_sw_r)
    ss_arr = np.array(ss_vals)
    sw_ss_arr = np.array(is_sw_ss)

    pos_r = r_arr[sw_arr == 1]
    neg_r = r_arr[sw_arr == 0]
    pos_ss = ss_arr[sw_ss_arr == 1] if len(ss_vals) else np.array([])
    neg_ss = ss_arr[sw_ss_arr == 0] if len(ss_vals) else np.array([])

    return _AccuracyMetrics(
        gene_driver_auc=_auc(r_arr, sw_arr),
        switch_strength_auc=_auc(ss_arr, sw_ss_arr),
        mean_r_switching=float(pos_r.mean()) if len(pos_r) else float("nan"),
        mean_r_nonswitching=float(neg_r.mean()) if len(neg_r) else float("nan"),
        median_ss_switching=float(np.median(pos_ss)) if len(pos_ss) else float("nan"),
        median_ss_nonswitching=float(np.median(neg_ss)) if len(neg_ss) else float("nan"),
        n_fitted_modules=len(results),
        n_genes_in_modules=sum(len(r.gene_driver_table) for r in results.values()),
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures (fit + explain each bundle once)
# ---------------------------------------------------------------------------

def _make_explain_fixture(spec):
    """Return a pytest session-scoped fixture that builds, fits, and explains a bundle."""
    @pytest.fixture(scope="module")
    def _fixture():
        bundle = _make_bundle(spec)
        truth_switch = bundle.feature_tables["truth_switch"]
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            module_table = _fit_and_write_artifacts(bundle, artifact_dir)
            if module_table.empty:
                pytest.skip(f"Baseline found no modules for {spec.name}.")
            results = _run_explain(bundle, artifact_dir)
        metrics = _compute_accuracy(results, truth_switch)
        return metrics
    return _fixture


noisy_metrics = _make_explain_fixture(_NOISY_SPEC)
nonlinear_metrics = _make_explain_fixture(_NONLINEAR_SPEC)
xlarge_mini_metrics = _make_explain_fixture(_XLARGE_MINI_SPEC)
xxlarge_mini_metrics = _make_explain_fixture(_XXLARGE_MINI_SPEC)


# ---------------------------------------------------------------------------
# Tests — noisy_v1
# ---------------------------------------------------------------------------

class TestNoisyAccuracy:
    """noisy_v1: high NB dispersion (15×), strong confounder (0.6), 8 small power-law modules."""

    def test_gene_driver_auc(self, noisy_metrics):
        m = noisy_metrics
        assert m.gene_driver_auc >= 0.60, (
            f"noisy gene driver AUC={m.gene_driver_auc:.3f} < 0.60"
        )

    def test_mean_r_direction(self, noisy_metrics):
        m = noisy_metrics
        assert m.mean_r_switching > m.mean_r_nonswitching, (
            f"noisy mean |r|: switching={m.mean_r_switching:.3f} not > "
            f"non-switching={m.mean_r_nonswitching:.3f}"
        )

    def test_switch_strength_auc(self, noisy_metrics):
        m = noisy_metrics
        assert m.switch_strength_auc >= 0.55, (
            f"noisy switch_strength AUC={m.switch_strength_auc:.3f} < 0.55"
        )

    def test_switch_strength_direction(self, noisy_metrics):
        m = noisy_metrics
        assert m.median_ss_switching > m.median_ss_nonswitching, (
            f"noisy median switch_strength: switching={m.median_ss_switching:.3f} not > "
            f"non-switching={m.median_ss_nonswitching:.3f}"
        )


# ---------------------------------------------------------------------------
# Tests — nonlinear_v1
# ---------------------------------------------------------------------------

class TestNonlinearAccuracy:
    """nonlinear_v1: radial/product module activation (bimodal latent, not linear)."""

    def test_gene_driver_auc(self, nonlinear_metrics):
        m = nonlinear_metrics
        assert m.gene_driver_auc >= 0.60, (
            f"nonlinear gene driver AUC={m.gene_driver_auc:.3f} < 0.60"
        )

    def test_mean_r_direction(self, nonlinear_metrics):
        m = nonlinear_metrics
        assert m.mean_r_switching > m.mean_r_nonswitching, (
            f"nonlinear mean |r|: switching={m.mean_r_switching:.3f} not > "
            f"non-switching={m.mean_r_nonswitching:.3f}"
        )

    def test_switch_strength_auc(self, nonlinear_metrics):
        m = nonlinear_metrics
        assert m.switch_strength_auc >= 0.55, (
            f"nonlinear switch_strength AUC={m.switch_strength_auc:.3f} < 0.55"
        )

    def test_switch_strength_direction(self, nonlinear_metrics):
        m = nonlinear_metrics
        assert m.median_ss_switching > m.median_ss_nonswitching, (
            f"nonlinear median switch_strength: switching={m.median_ss_switching:.3f} not > "
            f"non-switching={m.median_ss_nonswitching:.3f}"
        )


# ---------------------------------------------------------------------------
# Tests — xlarge_mini (xlarge_v1 structural params at 1/10 gene scale)
# ---------------------------------------------------------------------------

class TestXlargeMiniAccuracy:
    """xlarge_mini: 12 modules, ~14% switching, dispersion=7, confounder=0.4."""

    def test_gene_driver_auc(self, xlarge_mini_metrics):
        m = xlarge_mini_metrics
        assert m.gene_driver_auc >= 0.60, (
            f"xlarge_mini gene driver AUC={m.gene_driver_auc:.3f} < 0.60"
        )

    def test_mean_r_direction(self, xlarge_mini_metrics):
        m = xlarge_mini_metrics
        assert m.mean_r_switching > m.mean_r_nonswitching, (
            f"xlarge_mini mean |r|: switching={m.mean_r_switching:.3f} not > "
            f"non-switching={m.mean_r_nonswitching:.3f}"
        )

    def test_switch_strength_direction(self, xlarge_mini_metrics):
        m = xlarge_mini_metrics
        assert m.median_ss_switching > m.median_ss_nonswitching, (
            f"xlarge_mini median switch_strength: switching={m.median_ss_switching:.3f} not > "
            f"non-switching={m.median_ss_nonswitching:.3f}"
        )


# ---------------------------------------------------------------------------
# Tests — xxlarge_mini (xxlarge_v1 structural params at 1/10 gene scale)
# ---------------------------------------------------------------------------

class TestXxlargeMiniAccuracy:
    """xxlarge_mini: 16 modules, 24% switching, dispersion=7, confounder=0.4."""

    def test_gene_driver_auc(self, xxlarge_mini_metrics):
        m = xxlarge_mini_metrics
        assert m.gene_driver_auc >= 0.60, (
            f"xxlarge_mini gene driver AUC={m.gene_driver_auc:.3f} < 0.60"
        )

    def test_mean_r_direction(self, xxlarge_mini_metrics):
        m = xxlarge_mini_metrics
        assert m.mean_r_switching > m.mean_r_nonswitching, (
            f"xxlarge_mini mean |r|: switching={m.mean_r_switching:.3f} not > "
            f"non-switching={m.mean_r_nonswitching:.3f}"
        )

    def test_switch_strength_direction(self, xxlarge_mini_metrics):
        m = xxlarge_mini_metrics
        assert m.median_ss_switching > m.median_ss_nonswitching, (
            f"xxlarge_mini median switch_strength: switching={m.median_ss_switching:.3f} not > "
            f"non-switching={m.median_ss_nonswitching:.3f}"
        )
