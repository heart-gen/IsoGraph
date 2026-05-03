"""Annotation and consequence integration (Stage 8C)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from isograph.explain.structure import STRUCTURAL_LABELS, _BOOLEAN_LABELS, _NUMERIC_LABELS


# Canonical annotation column names
_BOOLEAN_ANNOTATION_COLUMNS: list[str] = _BOOLEAN_LABELS
_NUMERIC_ANNOTATION_COLUMNS: list[str] = _NUMERIC_LABELS
_ALL_ANNOTATION_COLUMNS: list[str] = STRUCTURAL_LABELS

# Alias map: lowercase input column name → canonical name
_ALIAS_MAP: dict[str, str] = {
    # Native Stage 8C structural labels
    "first_exon_changed": "first_exon_changed",
    "last_exon_changed": "last_exon_changed",
    "internal_exon_difference": "internal_exon_difference",
    "cds_changed": "cds_changed",
    "utr_changed": "utr_changed",
    "biotype_switch": "biotype_switch",
    "coding_status_change": "coding_status_change",
    "transcript_length_delta": "transcript_length_delta",
    "cds_length_delta": "cds_length_delta",
    "shared_exon_fraction": "shared_exon_fraction",
    # ISA / external tool interoperability aliases
    "alt_first_last_exon": "first_exon_changed",
    "cassette_exon": "internal_exon_difference",
    "exon_skipping": "internal_exon_difference",
    "intron_retention": "internal_exon_difference",
    "coding_change": "cds_changed",
    "coding_noncoding_switch": "coding_status_change",
    "coding_region_change": "cds_changed",
}

# Recognized feature ID column names (checked in order)
_FEATURE_ID_ALIASES: tuple[str, ...] = ("feature_id", "transcript_id", "isoform_id")


def _coerce_bool_column(series: pd.Series) -> pd.array:
    """Coerce a series to pd.BooleanDtype, handling string TRUE/FALSE."""
    if hasattr(series, "dtype") and isinstance(series.dtype, pd.BooleanDtype):
        return series
    if pd.api.types.is_bool_dtype(series):
        return pd.array(series.tolist(), dtype=pd.BooleanDtype())
    if pd.api.types.is_numeric_dtype(series):
        vals = series.map(lambda x: None if pd.isna(x) else bool(x))
        return pd.array(vals.tolist(), dtype=pd.BooleanDtype())
    # String coercion
    mapped = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
    )
    return pd.array(mapped.tolist(), dtype=pd.BooleanDtype())


def load_annotation_table(source: pd.DataFrame | Path | str) -> pd.DataFrame:
    """Load and normalize a structural annotation table.

    Accepts a DataFrame or file path (TSV, CSV, CSV.gz, or Parquet). Normalizes
    column aliases to canonical annotation names and coerces boolean columns to
    pd.BooleanDtype().

    Args:
        source: In-memory DataFrame or path to a TSV/CSV/Parquet file.

    Returns:
        Normalized DataFrame with feature_id column and any recognized
        annotation columns.

    Raises:
        FileNotFoundError: If source is a non-existent path.
        ValueError: If no feature_id-compatible column is found.
    """
    if not isinstance(source, pd.DataFrame):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Annotation table not found: {path}")
        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        elif suffix in (".tsv",) or path.name.endswith(".tsv.gz"):
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
    else:
        df = source.copy()

    df = df.copy()

    # Detect and rename feature_id column
    feat_col: str | None = None
    for alias in _FEATURE_ID_ALIASES:
        if alias in df.columns:
            feat_col = alias
            break
    if feat_col is None:
        raise ValueError(
            f"annotation_table must contain a feature_id column. "
            f"Recognized names: {_FEATURE_ID_ALIASES}. "
            f"Found columns: {list(df.columns)}"
        )
    if feat_col != "feature_id":
        df = df.rename(columns={feat_col: "feature_id"})

    # Rename alias columns to canonical names (case-insensitive)
    rename_map: dict[str, str] = {}
    for col in list(df.columns):
        canonical = _ALIAS_MAP.get(col.strip().lower())
        if canonical and col != canonical:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    # Coerce boolean annotation columns
    for col in _BOOLEAN_ANNOTATION_COLUMNS:
        if col in df.columns:
            df[col] = _coerce_bool_column(df[col])

    # Warn if no annotation columns detected
    detected = [c for c in _ALL_ANNOTATION_COLUMNS if c in df.columns]
    if not detected:
        warnings.warn(
            "annotation_table contains no recognized annotation columns. "
            f"Expected one of: {_ALL_ANNOTATION_COLUMNS}",
            UserWarning,
            stacklevel=2,
        )

    return df


def annotate_driver_table(
    gene_driver_table: pd.DataFrame,
    annotation_table: pd.DataFrame,
) -> pd.DataFrame:
    """Merge gene-level annotation flags into gene_driver_table.

    For each gene, boolean flags are True if ANY transcript has that label.
    Numeric metrics are the max across transcripts.
    Unmatched genes get pd.NA / NaN.

    Args:
        gene_driver_table: Output of compute_gene_driver_table; must have gene_id column.
        annotation_table: Normalized table from load_annotation_table; must have gene_id.

    Returns:
        Copy of gene_driver_table with annotation columns appended.
    """
    annot_cols = [c for c in _ALL_ANNOTATION_COLUMNS if c in annotation_table.columns]
    if not annot_cols:
        warnings.warn(
            "annotation_table has no recognized annotation columns; annotate_driver_table is a no-op.",
            UserWarning,
            stacklevel=2,
        )
        return gene_driver_table.copy()

    if "gene_id" not in annotation_table.columns:
        raise ValueError(
            "annotation_table must contain a gene_id column for gene-level aggregation."
        )

    bool_cols = [c for c in annot_cols if c in _BOOLEAN_ANNOTATION_COLUMNS]
    num_cols = [c for c in annot_cols if c in _NUMERIC_ANNOTATION_COLUMNS]

    agg_parts: list[pd.DataFrame] = []
    if bool_cols:
        bool_agg = annotation_table.groupby("gene_id")[bool_cols].any(skipna=True)
        for col in bool_cols:
            bool_agg[col] = pd.array(bool_agg[col].tolist(), dtype=pd.BooleanDtype())
        agg_parts.append(bool_agg)
    if num_cols:
        num_agg = annotation_table.groupby("gene_id")[num_cols].max()
        agg_parts.append(num_agg)

    if len(agg_parts) == 2:
        gene_agg = agg_parts[0].join(agg_parts[1], how="outer")
    else:
        gene_agg = agg_parts[0]
    gene_agg = gene_agg.reset_index()

    return gene_driver_table.merge(gene_agg, on="gene_id", how="left")


def annotate_transcript_table(
    transcript_polarity_table: pd.DataFrame,
    annotation_table: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-transcript annotation flags into transcript_polarity_table.

    Left-joins on feature_id. Unmatched features get pd.NA / NaN.
    Empty input table is returned unchanged.

    Args:
        transcript_polarity_table: Output of compute_transcript_polarity_table;
            must have feature_id column.
        annotation_table: Normalized table from load_annotation_table;
            must have feature_id column.

    Returns:
        Copy of transcript_polarity_table with annotation columns appended.
    """
    if transcript_polarity_table.empty:
        return transcript_polarity_table.copy()

    annot_cols = [c for c in _ALL_ANNOTATION_COLUMNS if c in annotation_table.columns]
    if not annot_cols:
        warnings.warn(
            "annotation_table has no recognized annotation columns; annotate_transcript_table is a no-op.",
            UserWarning,
            stacklevel=2,
        )
        return transcript_polarity_table.copy()

    subset = annotation_table[["feature_id"] + annot_cols].copy()
    if subset["feature_id"].duplicated().any():
        n_dupes = int(subset["feature_id"].duplicated().sum())
        warnings.warn(
            f"annotation_table has {n_dupes} duplicate feature_id(s); keeping first occurrence.",
            UserWarning,
            stacklevel=2,
        )
        subset = subset.drop_duplicates(subset="feature_id", keep="first")

    return transcript_polarity_table.merge(subset, on="feature_id", how="left")
