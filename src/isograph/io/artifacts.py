"""Dataset artifact I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from isograph.utils import ensure_dir, write_json
from isograph.validation import DatasetManifest, FeatureTableSpec, LoadedDataset, MatrixSpec


@dataclass
class DatasetBundle:
    manifest: DatasetManifest
    sample_table: pd.DataFrame
    feature_tables: dict[str, pd.DataFrame]
    matrices: dict[str, np.ndarray]
    truth_tables: dict[str, pd.DataFrame]


def save_dense_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savez_compressed(path, data=matrix)


def load_dense_matrix(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        return loaded["data"]


def save_dataset_bundle(bundle: DatasetBundle, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    bundle.sample_table.to_parquet(output_dir / bundle.manifest.sample_table, index=False)

    for spec in bundle.manifest.feature_tables:
        table = bundle.feature_tables[spec.kind]
        table.to_parquet(output_dir / spec.filename, index=False)

    for spec in bundle.manifest.matrices:
        save_dense_matrix(output_dir / spec.filename, bundle.matrices[spec.assay_name])

    for truth_name, table in bundle.truth_tables.items():
        table.to_parquet(output_dir / truth_name, index=False)

    write_json(output_dir / "manifest.json", bundle.manifest.model_dump())
    return output_dir


def load_dataset_bundle(dataset_dir: Path) -> DatasetBundle:
    manifest = DatasetManifest.model_validate_json((dataset_dir / "manifest.json").read_text())
    sample_table = pd.read_parquet(dataset_dir / manifest.sample_table)
    feature_tables = {
        spec.kind: pd.read_parquet(dataset_dir / spec.filename) for spec in manifest.feature_tables
    }
    matrices = {
        spec.assay_name: load_dense_matrix(dataset_dir / spec.filename) for spec in manifest.matrices
    }
    truth_tables = {name: pd.read_parquet(dataset_dir / name) for name in manifest.truth_tables}
    return DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables=feature_tables,
        matrices=matrices,
        truth_tables=truth_tables,
    )


def describe_dataset(dataset_dir: Path) -> LoadedDataset:
    bundle = load_dataset_bundle(dataset_dir)
    gene_table = bundle.feature_tables.get("gene")
    return LoadedDataset(
        manifest_path=dataset_dir / "manifest.json",
        n_samples=len(bundle.sample_table),
        n_genes=0 if gene_table is None else len(gene_table),
        available_assays=sorted(bundle.matrices),
    )


def validate_bundle_inputs(
    sample_table: pd.DataFrame,
    transcript_table: pd.DataFrame,
    transcript_counts: np.ndarray,
    gene_table: pd.DataFrame | None = None,
    gene_counts: np.ndarray | None = None,
) -> None:
    """Validate raw inputs before building a dataset bundle.

    Checks alignment rules, required columns, and data integrity. Raises
    ``ValueError`` listing all violations found (not just the first).

    Parameters
    ----------
    sample_table:
        One row per sample. Must contain ``sample_id``. Column order must match
        the column order of every matrix.
    transcript_table:
        One row per transcript. Must contain ``transcript_id`` and ``gene_id``.
        Row order must match the row order of ``transcript_counts``.
    transcript_counts:
        Dense count matrix of shape ``(n_transcripts, n_samples)``.
    gene_table:
        Optional. One row per gene. Must contain ``gene_id`` when provided.
        Row order must match the row order of ``gene_counts``.
    gene_counts:
        Optional. Dense count matrix of shape ``(n_genes, n_samples)``.
        Required when ``gene_table`` is provided, and vice versa.
    """
    errors: list[str] = []
    n_samples = len(sample_table)

    # ── sample_table ─────────────────────────────────────────────────────────
    if "sample_id" not in sample_table.columns:
        errors.append("sample_table is missing required column 'sample_id'")
    elif sample_table["sample_id"].duplicated().any():
        dupes = sample_table["sample_id"][sample_table["sample_id"].duplicated()].tolist()
        errors.append(f"sample_table has duplicate sample_id values: {dupes}")

    # ── transcript_table ──────────────────────────────────────────────────────
    for col in ("transcript_id", "gene_id"):
        if col not in transcript_table.columns:
            errors.append(f"transcript_table is missing required column '{col}'")
    if "transcript_id" in transcript_table.columns and transcript_table["transcript_id"].duplicated().any():
        errors.append("transcript_table has duplicate transcript_id values")

    # ── transcript_counts shape ───────────────────────────────────────────────
    if transcript_counts.ndim != 2:
        errors.append(
            f"transcript_counts must be 2-D, got shape {transcript_counts.shape}"
        )
    else:
        n_tx, n_col = transcript_counts.shape
        if n_tx != len(transcript_table):
            errors.append(
                f"transcript_counts rows ({n_tx}) != transcript_table rows "
                f"({len(transcript_table)})"
            )
        if n_col != n_samples:
            errors.append(
                f"transcript_counts columns ({n_col}) != sample_table rows "
                f"({n_samples})"
            )

    # ── transcript_counts values ──────────────────────────────────────────────
    if transcript_counts.ndim == 2:
        if np.any(np.isnan(transcript_counts)):
            errors.append("transcript_counts contains NaN values")
        if np.any(np.isinf(transcript_counts)):
            errors.append("transcript_counts contains Inf values")
        if np.any(transcript_counts < 0):
            errors.append("transcript_counts contains negative values")

    # ── gene_table / gene_counts consistency ──────────────────────────────────
    if (gene_table is None) != (gene_counts is None):
        errors.append(
            "gene_table and gene_counts must both be provided or both be None"
        )
    if gene_table is not None and gene_counts is not None:
        if "gene_id" not in gene_table.columns:
            errors.append("gene_table is missing required column 'gene_id'")
        elif gene_table["gene_id"].duplicated().any():
            errors.append("gene_table has duplicate gene_id values")

        if gene_counts.ndim != 2:
            errors.append(f"gene_counts must be 2-D, got shape {gene_counts.shape}")
        else:
            n_g, n_col = gene_counts.shape
            if n_g != len(gene_table):
                errors.append(
                    f"gene_counts rows ({n_g}) != gene_table rows ({len(gene_table)})"
                )
            if n_col != n_samples:
                errors.append(
                    f"gene_counts columns ({n_col}) != sample_table rows ({n_samples})"
                )

        if gene_counts.ndim == 2:
            if np.any(np.isnan(gene_counts)):
                errors.append("gene_counts contains NaN values")
            if np.any(np.isinf(gene_counts)):
                errors.append("gene_counts contains Inf values")
            if np.any(gene_counts < 0):
                errors.append("gene_counts contains negative values")

        # gene_id cross-reference
        if (
            gene_table is not None
            and "gene_id" in gene_table.columns
            and "gene_id" in transcript_table.columns
        ):
            tx_genes = set(transcript_table["gene_id"].unique())
            known_genes = set(gene_table["gene_id"].unique())
            orphans = tx_genes - known_genes
            if orphans:
                n_shown = min(5, len(orphans))
                sample = sorted(orphans)[:n_shown]
                errors.append(
                    f"transcript_table references {len(orphans)} gene_id value(s) not in "
                    f"gene_table (first {n_shown}: {sample})"
                )

    if errors:
        bullet_list = "\n  - ".join(errors)
        raise ValueError(
            f"validate_bundle_inputs found {len(errors)} problem(s):\n  - {bullet_list}"
        )


def build_feature_spec(kind: str, filename: str, table: pd.DataFrame) -> FeatureTableSpec:
    return FeatureTableSpec(kind=kind, filename=filename, n_rows=len(table))


def build_matrix_spec(name: str, filename: str, matrix: np.ndarray) -> MatrixSpec:
    return MatrixSpec(
        assay_name=name,
        filename=filename,
        n_features=int(matrix.shape[0]),
        n_samples=int(matrix.shape[1]),
    )
