"""Input loading and validation for module explanation (Stage 7A)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


_REQUIRED_FEATURE_META_COLS = ["feature_id", "gene_id", "feature_type"]
_OPTIONAL_FEATURE_META_COLS = ["gene_name", "transcript_id", "exon_id", "event_id", "source_coordinate"]
_MIN_OVERLAP_SAMPLES = 10


def _check_required_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _normalize_sample_index(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Ensure df.index contains sample IDs. Accepts a 'sample_id' column or existing index."""
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].tolist()
        raise ValueError(f"{context} has duplicate sample IDs: {dupes[:5]}")
    return df


def load_explain_inputs(
    artifact_dir: Path,
    feature_table: pd.DataFrame,
    feature_meta: pd.DataFrame,
    module_ids: list[str] | None,
    module_score_table: pd.DataFrame | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    pd.DataFrame | None,
]:
    """Load and validate all inputs for explain_module.

    Returns:
        modules, feature_scores, feature_table_aligned, feature_meta_validated,
        resolved_module_ids, sample_ids, module_score_table_aligned
    """
    # --- Load artifact files ---
    modules_path = artifact_dir / "modules.parquet"
    scores_path = artifact_dir / "feature_scores.parquet"
    for path in (modules_path, scores_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact file not found: {path}. "
                f"Run 'isograph fit' to generate it."
            )
    modules = pd.read_parquet(modules_path)
    feature_scores = pd.read_parquet(scores_path)

    # --- Validate feature_meta ---
    _check_required_columns(feature_meta, _REQUIRED_FEATURE_META_COLS, "feature_meta")
    absent_optional = [c for c in _OPTIONAL_FEATURE_META_COLS if c not in feature_meta.columns]
    if absent_optional:
        warnings.warn(
            f"feature_meta is missing optional columns (annotations will be limited): {absent_optional}",
            UserWarning,
            stacklevel=3,
        )
    if feature_meta["feature_id"].duplicated().any():
        dupes = feature_meta["feature_id"][feature_meta["feature_id"].duplicated()].tolist()
        raise ValueError(f"feature_meta has duplicate feature_id values: {dupes[:5]}")

    # --- Normalize feature_table sample index ---
    feature_table = _normalize_sample_index(feature_table.copy(), "feature_table")

    # --- Extract sample IDs from feature_scores ---
    score_sample_ids = [c for c in feature_scores.columns if c != "gene_id"]

    # --- Sample alignment ---
    overlap = sorted(set(score_sample_ids) & set(feature_table.index.astype(str)))
    if len(overlap) < _MIN_OVERLAP_SAMPLES:
        raise ValueError(
            f"Only {len(overlap)} samples overlap between feature_scores and feature_table "
            f"(minimum required: {_MIN_OVERLAP_SAMPLES}). Check that sample IDs match."
        )
    dropped_scores = set(score_sample_ids) - set(overlap)
    dropped_table = set(feature_table.index.astype(str)) - set(overlap)
    if dropped_scores:
        warnings.warn(
            f"{len(dropped_scores)} sample(s) in feature_scores not in feature_table and will be dropped.",
            UserWarning,
            stacklevel=3,
        )
    if dropped_table:
        warnings.warn(
            f"{len(dropped_table)} sample(s) in feature_table not in feature_scores and will be dropped.",
            UserWarning,
            stacklevel=3,
        )

    sample_ids = overlap
    # Align feature_scores columns (keep gene_id + sample_ids in order)
    feature_scores_aligned = feature_scores[["gene_id"] + sample_ids].copy()
    feature_table_aligned = feature_table.loc[sample_ids].copy()

    # --- Resolve module IDs ---
    all_module_ids = sorted(modules["module_id"].unique())
    if module_ids is None:
        resolved_module_ids = all_module_ids
    else:
        unknown = [m for m in module_ids if m not in all_module_ids]
        if unknown:
            raise ValueError(
                f"Unknown module ID(s): {unknown}. "
                f"Available modules: {all_module_ids}"
            )
        resolved_module_ids = list(module_ids)

    # --- Align module_score_table if provided ---
    aligned_mst: pd.DataFrame | None = None
    if module_score_table is not None:
        mst = _normalize_sample_index(module_score_table.copy(), "module_score_table")
        missing_cols = [m for m in resolved_module_ids if m not in mst.columns]
        if missing_cols:
            raise ValueError(
                f"module_score_table is missing columns for requested module(s): {missing_cols}"
            )
        mst_overlap = sorted(set(sample_ids) & set(mst.index.astype(str)))
        if len(mst_overlap) < _MIN_OVERLAP_SAMPLES:
            raise ValueError(
                f"Only {len(mst_overlap)} samples overlap between module_score_table and the "
                f"feature data (minimum required: {_MIN_OVERLAP_SAMPLES})."
            )
        aligned_mst = mst.loc[sample_ids, resolved_module_ids]

    return (
        modules,
        feature_scores_aligned,
        feature_table_aligned,
        feature_meta,
        resolved_module_ids,
        sample_ids,
        aligned_mst,
    )
