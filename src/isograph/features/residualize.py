"""Residualization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_design_matrix(sample_table: pd.DataFrame, covariates: list[str]) -> np.ndarray:
    design_parts = [np.ones((len(sample_table), 1), dtype=float)]
    for column in covariates:
        if column not in sample_table.columns:
            continue
        series = sample_table[column]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=float(np.nanmean(values) if np.isnan(values).any() else 0.0))
            design_parts.append(values[:, None])
        else:
            dummies = pd.get_dummies(series.fillna("missing"), prefix=column, dtype=float)
            if not dummies.empty:
                design_parts.append(dummies.to_numpy(dtype=float))
    return np.hstack(design_parts)


def residualize_rows(matrix: np.ndarray, design: np.ndarray) -> np.ndarray:
    if design.shape[1] == 1:
        return matrix - matrix.mean(axis=1, keepdims=True)
    beta, *_ = np.linalg.lstsq(design, matrix.T, rcond=None)
    fitted = design @ beta
    return matrix - fitted.T
