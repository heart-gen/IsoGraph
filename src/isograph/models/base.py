"""Model interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FitArtifacts:
    module_table: pd.DataFrame
    edge_table: pd.DataFrame
    trait_table: pd.DataFrame
    feature_scores: pd.DataFrame
    calibration: dict | None = None
    checkpoint_path: Path | None = None
    eigengene_table: pd.DataFrame | None = None


def compute_trait_associations(
    module_table: pd.DataFrame,
    feature_scores: pd.DataFrame,
    sample_table: pd.DataFrame,
    trait_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute eigengene-trait associations for all modules.

    Numeric columns → Pearson correlation (effect = r).
    Binary categorical columns (exactly 2 unique values) → Welch t-test
    (effect = group1_mean - group0_mean, groups ordered by sorted category label).
    Columns absent from sample_table or with >2 categories are skipped.
    """
    sample_ids = (
        sample_table["sample_id"].tolist()
        if "sample_id" in sample_table.columns
        else list(range(len(sample_table)))
    )
    if module_table.empty:
        return (
            pd.DataFrame(columns=["module_id", "trait", "effect", "pvalue"]),
            pd.DataFrame(columns=["module_id"] + sample_ids),
        )
    rows = []
    eigengene_rows: dict[str, list] = {}
    for module_id in sorted(module_table["module_id"].unique()):
        genes = module_table.loc[module_table["module_id"] == module_id, "gene_id"]
        subset = feature_scores.loc[feature_scores["gene_id"].isin(genes)]
        eigengene = subset.drop(columns=["gene_id"]).to_numpy(dtype=float).mean(axis=0)
        eigengene_rows[module_id] = eigengene.tolist()
        for col in trait_columns:
            if col not in sample_table.columns:
                continue
            series = sample_table[col]
            if pd.api.types.is_numeric_dtype(series):
                vals = series.to_numpy(dtype=float)
                mask = np.isfinite(vals) & np.isfinite(eigengene)
                if mask.sum() < 10:
                    continue
                effect, pvalue = stats.pearsonr(eigengene[mask], vals[mask])
                rows.append({"module_id": module_id, "trait": col, "effect": float(effect), "pvalue": float(pvalue)})
            else:
                cats = sorted(series.dropna().unique())
                if len(cats) != 2:
                    continue
                cat0, cat1 = cats
                grp0 = eigengene[(series == cat0).to_numpy()]
                grp1 = eigengene[(series == cat1).to_numpy()]
                if len(grp0) < 2 or len(grp1) < 2:
                    continue
                ttest = stats.ttest_ind(grp0, grp1, equal_var=False)
                effect = float(grp1.mean() - grp0.mean())
                rows.append({"module_id": module_id, "trait": col, "effect": effect, "pvalue": float(ttest.pvalue)})
    eigengene_table = (
        pd.DataFrame(eigengene_rows, index=sample_ids)
        .T.reset_index()
        .rename(columns={"index": "module_id"})
    )
    return pd.DataFrame(rows), eigengene_table


class NetworkModel:
    def fit(self, *args, **kwargs) -> FitArtifacts:
        raise NotImplementedError
