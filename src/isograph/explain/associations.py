"""Pure statistical computations for module explanation (Stage 7A)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

from isograph.explain.config import ExplainConfig


def compute_eigengene(
    feature_scores: pd.DataFrame,
    module_genes: list[str],
    sample_ids: list[str],
    min_complete: int = 3,
) -> np.ndarray:
    """Compute module eigengene as the sign-aligned mean switch coordinate.

    Gene switch coordinates from gene_switch_coordinates() have per-gene arbitrary
    sign (from the SVD stable_sign convention). A naive mean can cancel to near-zero
    when signs are inconsistent across genes in the same module. This function
    bootstraps sign orientation from the first gene, then iterates to convergence.

    This is the correct formula for module explanation; compute_trait_associations
    in models/base.py uses the naive mean for historical reasons (it is only used
    for trait associations where sign inconsistency does not affect the result).
    """
    subset = feature_scores.loc[feature_scores["gene_id"].isin(module_genes)]
    if subset.empty:
        warnings.warn(
            "No genes from module overlap with feature_scores; eigengene will be all-NaN.",
            UserWarning,
            stacklevel=2,
        )
        return np.full(len(sample_ids), np.nan)
    matrix = subset[sample_ids].to_numpy(dtype=float)
    n_genes = len(matrix)
    if n_genes == 1:
        return matrix[0]

    # Bootstrap sign orientations against the first gene
    signs = np.ones(n_genes)
    ref = matrix[0]
    for i in range(1, n_genes):
        mask = np.isfinite(matrix[i]) & np.isfinite(ref)
        if mask.sum() >= min_complete:
            r = np.corrcoef(matrix[i][mask], ref[mask])[0, 1]
            if r < 0:
                signs[i] = -1.0

    # Iterate to refine against the current mean
    for _ in range(5):
        eigengene = np.nanmean(matrix * signs[:, None], axis=0)
        new_signs = signs.copy()
        for i in range(n_genes):
            mask = np.isfinite(matrix[i]) & np.isfinite(eigengene)
            if mask.sum() >= min_complete:
                r = np.corrcoef(matrix[i][mask], eigengene[mask])[0, 1]
                new_signs[i] = 1.0 if r >= 0 else -1.0
        if np.all(new_signs == signs):
            break
        signs = new_signs

    return np.nanmean(matrix * signs[:, None], axis=0)


def pearson_r_with_missing(
    x: np.ndarray,
    y: np.ndarray,
    min_complete: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pearson r for each row of x against y, handling missing values.

    Returns (r, pvalue, n_complete, missing_fraction).
    Features with fewer than min_complete finite pairs get NaN statistics.
    """
    n_features, n_samples = x.shape
    r_out = np.full(n_features, np.nan)
    p_out = np.full(n_features, np.nan)
    n_out = np.zeros(n_features, dtype=int)
    for i in range(n_features):
        mask = np.isfinite(x[i]) & np.isfinite(y)
        n_out[i] = int(mask.sum())
        if n_out[i] < min_complete:
            continue
        r_out[i], p_out[i] = stats.pearsonr(x[i][mask], y[mask])
    missing_frac = 1.0 - n_out / n_samples
    return r_out, p_out, n_out, missing_frac


def apply_fdr(pvalues: np.ndarray, method: str = "bh") -> np.ndarray:
    """Apply BH FDR correction, preserving NaN positions."""
    finite_mask = np.isfinite(pvalues)
    qvalues = np.full_like(pvalues, np.nan)
    if finite_mask.sum() > 0:
        qvalues[finite_mask] = false_discovery_control(pvalues[finite_mask], method=method)
    return qvalues


def compute_gene_driver_table(
    feature_scores: pd.DataFrame,
    eigengene: np.ndarray,
    module_genes: list[str],
    sample_ids: list[str],
    config: ExplainConfig,
) -> pd.DataFrame:
    """Rank module genes by Pearson r with the module eigengene.

    Columns: gene_id, r, pvalue, qvalue, n_samples, missing_fraction.
    Sorted descending by |r|.
    """
    _empty = pd.DataFrame(columns=["gene_id", "r", "pvalue", "qvalue", "n_samples", "missing_fraction"])
    subset = feature_scores.loc[feature_scores["gene_id"].isin(module_genes)].copy()
    if subset.empty:
        return _empty
    gene_ids = subset["gene_id"].tolist()
    matrix = subset[sample_ids].to_numpy(dtype=float)
    r, pvalue, n_complete, missing_frac = pearson_r_with_missing(matrix, eigengene, config.min_complete_pairs)
    qvalue = apply_fdr(pvalue, method=config.fdr_method)
    df = pd.DataFrame({
        "gene_id": gene_ids,
        "r": r,
        "pvalue": pvalue,
        "qvalue": qvalue,
        "n_samples": n_complete,
        "missing_fraction": missing_frac,
    })
    sort_key = np.where(np.isfinite(r), np.abs(r), 0.0)
    df = df.iloc[np.argsort(-sort_key)].reset_index(drop=True)
    return df


def compute_transcript_polarity_table(
    feature_table: pd.DataFrame,
    feature_meta: pd.DataFrame,
    eigengene: np.ndarray,
    sample_ids: list[str],
    module_genes: list[str],
    config: ExplainConfig,
) -> pd.DataFrame:
    """Correlate transcript-usage features with the module eigengene.

    Filters to features where gene_id is in module_genes and feature_type matches
    config.transcript_usage_feature_type. switch_strength per gene captures the
    within-gene isoform contrast: max(r) - min(r) across that gene's transcripts.

    Columns: feature_id, gene_id, [transcript_id], r, pvalue, qvalue,
             n_samples, missing_fraction, switch_strength.
    """
    has_transcript_id = "transcript_id" in feature_meta.columns
    base_cols = ["feature_id", "gene_id"]
    if has_transcript_id:
        base_cols.append("transcript_id")
    empty_cols = base_cols + ["r", "pvalue", "qvalue", "n_samples", "missing_fraction", "switch_strength"]

    rel_mask = (
        feature_meta["gene_id"].isin(module_genes)
        & (feature_meta["feature_type"] == config.transcript_usage_feature_type)
    )
    rel_meta = feature_meta[rel_mask].copy()
    available = [f for f in rel_meta["feature_id"] if f in feature_table.columns]
    if not available:
        return pd.DataFrame(columns=empty_cols)

    rel_meta = rel_meta[rel_meta["feature_id"].isin(available)].reset_index(drop=True)
    feature_ids = rel_meta["feature_id"].tolist()

    # shape: (n_features, n_samples)
    matrix = feature_table.loc[sample_ids, feature_ids].to_numpy(dtype=float).T
    r, pvalue, n_complete, missing_frac = pearson_r_with_missing(matrix, eigengene, config.min_complete_pairs)
    qvalue = apply_fdr(pvalue, method=config.fdr_method)

    # switch_strength = max(r) - min(r) per gene
    r_series = pd.Series(r, index=feature_ids)
    gene_switch: dict[str, float] = {}
    for gene_id, grp in rel_meta.groupby("gene_id"):
        gene_r = [r_series[fid] for fid in grp["feature_id"] if np.isfinite(r_series[fid])]
        gene_switch[str(gene_id)] = float(max(gene_r) - min(gene_r)) if len(gene_r) >= 2 else 0.0
    switch_strength = rel_meta["gene_id"].map(gene_switch).to_numpy(dtype=float)

    df = rel_meta[base_cols].copy()
    df["r"] = r
    df["pvalue"] = pvalue
    df["qvalue"] = qvalue
    df["n_samples"] = n_complete
    df["missing_fraction"] = missing_frac
    df["switch_strength"] = switch_strength
    return df.reset_index(drop=True)


def compute_high_vs_low_table(
    feature_table: pd.DataFrame,
    feature_meta: pd.DataFrame,
    eigengene: np.ndarray,
    sample_ids: list[str],
    config: ExplainConfig,
) -> pd.DataFrame:
    """Compare feature values between high-vs-low module score samples.

    Splits samples at config.split_percentile of the eigengene. Uses Welch t-test.
    No FDR correction — this table is descriptive; callers apply FDR as needed.

    Columns: feature_id, gene_id, mean_high, mean_low, delta, se,
             tstat, pvalue, n_high, n_low, missing_fraction.
    """
    empty_cols = [
        "feature_id", "gene_id", "mean_high", "mean_low", "delta",
        "se", "tstat", "pvalue", "n_high", "n_low", "missing_fraction",
    ]
    meta_features = set(feature_meta["feature_id"].tolist())
    available = [f for f in feature_table.columns if f in meta_features]
    if not available:
        return pd.DataFrame(columns=empty_cols)

    threshold = float(np.nanpercentile(eigengene, config.split_percentile))
    high_mask = eigengene >= threshold
    low_mask = ~high_mask

    meta_lookup = feature_meta.set_index("feature_id")["gene_id"]
    n_total = len(sample_ids)
    rows = []
    for feature_id in available:
        vals = feature_table.loc[sample_ids, feature_id].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        high_vals = vals[high_mask & finite]
        low_vals = vals[low_mask & finite]
        gene_id = meta_lookup.get(feature_id, np.nan)
        missing_frac = 1.0 - finite.sum() / n_total

        row: dict = {
            "feature_id": feature_id,
            "gene_id": gene_id,
            "missing_fraction": missing_frac,
            "n_high": len(high_vals),
            "n_low": len(low_vals),
        }
        if len(high_vals) < 2 or len(low_vals) < 2:
            row.update({"mean_high": np.nan, "mean_low": np.nan, "delta": np.nan,
                        "se": np.nan, "tstat": np.nan, "pvalue": np.nan})
        else:
            mean_h, mean_l = float(high_vals.mean()), float(low_vals.mean())
            se = float(np.sqrt(high_vals.var(ddof=1) / len(high_vals) + low_vals.var(ddof=1) / len(low_vals)))
            res = stats.ttest_ind(high_vals, low_vals, equal_var=False)
            row.update({
                "mean_high": mean_h, "mean_low": mean_l,
                "delta": mean_h - mean_l, "se": se,
                "tstat": float(res.statistic), "pvalue": float(res.pvalue),
            })
        rows.append(row)

    return pd.DataFrame(rows, columns=empty_cols)
