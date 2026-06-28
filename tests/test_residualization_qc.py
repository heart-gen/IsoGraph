from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from isograph.features.residualize import (
    build_design_matrix,
    residualization_qc,
    residualize_rows,
)


def test_build_design_matrix_warns_on_missing_covariate() -> None:
    st = pd.DataFrame({"rin": np.linspace(5.0, 9.0, 10)})
    with pytest.warns(RuntimeWarning, match="not found"):
        design = build_design_matrix(st, ["rin", "median_tin"])
    # The present covariate is still used (intercept + rin); the absent one is dropped.
    assert design.shape == (10, 2)


def test_build_design_matrix_warns_on_constant_covariate() -> None:
    st = pd.DataFrame({"rin": np.linspace(5.0, 9.0, 10), "batch": ["A"] * 10})
    with pytest.warns(RuntimeWarning, match="constant"):
        design = build_design_matrix(st, ["rin", "batch"])
    assert design.shape == (10, 2)


def _confounded_features(seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_features, n_samples = 12, 60
    covariate = rng.normal(0.0, 1.0, n_samples)
    loading = rng.uniform(0.5, 2.0, n_features)
    signal = rng.normal(0.0, 1.0, (n_features, n_samples))
    before = signal + loading[:, None] * covariate[None, :]
    return before, pd.DataFrame({"rin": covariate})


def test_residualization_qc_removes_confound_axis() -> None:
    before, sample_table = _confounded_features()
    design = build_design_matrix(sample_table, ["rin"])
    after = residualize_rows(before, design)
    qc = residualization_qc(before, after, design)

    # The injected covariate axis is detectable before and gone after.
    assert qc["confound_r2_before"].median() > 0.2
    assert qc["confound_r2_after"].max() < 1e-6
    # Variance is reduced (collateral cost) but not eliminated.
    assert (qc["var_retained_frac"] < 1.0).all()
    assert (qc["var_retained_frac"] > 0.0).all()


def test_residualization_qc_attaches_feature_ids() -> None:
    before, sample_table = _confounded_features()
    design = build_design_matrix(sample_table, ["rin"])
    after = residualize_rows(before, design)
    feature_info = pd.DataFrame(
        {
            "feature_id": [f"f{i}" for i in range(before.shape[0])],
            "gene_id": [f"g{i}" for i in range(before.shape[0])],
            "feature_type": ["switch"] * before.shape[0],
        }
    )
    qc = residualization_qc(before, after, design, feature_info)
    assert list(qc.columns[:3]) == ["feature_id", "gene_id", "feature_type"]
    assert len(qc) == before.shape[0]
