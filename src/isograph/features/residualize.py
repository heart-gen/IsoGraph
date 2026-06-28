"""Residualization helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def build_design_matrix(sample_table: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    design_parts = [np.ones((len(sample_table), 1), dtype=float)]
    missing: list[str] = []
    constant: list[str] = []
    for column in covariates:
        if column not in sample_table.columns:
            missing.append(column)
            continue
        series = sample_table[column]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=float(np.nanmean(values) if np.isnan(values).any() else 0.0))
            if np.ptp(values) == 0.0:
                constant.append(column)
                continue
            design_parts.append(values[:, None])
        else:
            dummies = pd.get_dummies(series.fillna("missing"), prefix=column, dtype=float)
            if dummies.shape[1] <= 1:
                constant.append(column)
                continue
            design_parts.append(dummies.to_numpy(dtype=float))
    if missing:
        warnings.warn(
            f"residualize_covariates not found in sample_table and ignored: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )
    if constant:
        warnings.warn(
            f"residualize_covariates dropped as constant (no variance to regress out): {constant}",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.hstack(design_parts)


def residualize_rows(matrix: np.ndarray, design: np.ndarray) -> np.ndarray:
    if design.shape[1] == 1:
        return matrix - matrix.mean(axis=1, keepdims=True)
    beta, *_ = np.linalg.lstsq(design, matrix.T, rcond=None)
    fitted = design @ beta
    return matrix - fitted.T


def _covariate_r2(matrix: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Per-row fraction of variance explained by the non-intercept covariates."""
    covariates = design[:, 1:]
    n_features = matrix.shape[0]
    if covariates.shape[1] == 0 or n_features == 0:
        return np.zeros(n_features, dtype=float)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    covariates = covariates - covariates.mean(axis=0, keepdims=True)
    beta, *_ = np.linalg.lstsq(covariates, centered.T, rcond=None)
    fitted = (covariates @ beta).T
    ss_total = np.einsum("ij,ij->i", centered, centered)
    ss_fitted = np.einsum("ij,ij->i", fitted, fitted)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_total > 1e-12, ss_fitted / ss_total, 0.0)
    return np.clip(r2, 0.0, 1.0)


def residualization_qc(
    before: np.ndarray,
    after: np.ndarray,
    design: np.ndarray,
    feature_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Before/after QC for a covariate-residualized feature matrix.

    Reports, per feature row, the variance retained after regressing the
    covariates out (``var_retained_frac``) and how much of that feature aligned
    with the covariate axis before vs. after (``confound_r2_before`` /
    ``confound_r2_after``). A working residualization drives ``confound_r2_after``
    toward zero while ``var_retained_frac`` quantifies the collateral signal it
    cost -- the diagnostic that is observable on real data, where module recovery
    is not. Rows carry ``feature_info`` identifiers when supplied.
    """
    var_before = before.var(axis=1)
    var_after = after.var(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        retained = np.where(var_before > 1e-12, var_after / var_before, np.nan)
    qc = pd.DataFrame(
        {
            "var_before": var_before,
            "var_after": var_after,
            "var_retained_frac": retained,
            "confound_r2_before": _covariate_r2(before, design),
            "confound_r2_after": _covariate_r2(after, design),
        }
    )
    if feature_info is not None and len(feature_info) == len(qc):
        ids = [c for c in ("feature_id", "gene_id", "feature_type") if c in feature_info.columns]
        if ids:
            qc = pd.concat([feature_info[ids].reset_index(drop=True), qc], axis=1)
    return qc
