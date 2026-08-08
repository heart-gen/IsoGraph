"""Gene-level feature channel construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from isograph.features.switch import gene_switch_coordinates

FEATURE_SCORE_METADATA_COLUMNS = ("feature_id", "gene_id", "feature_type", "n_transcripts")


def feature_sample_columns(feature_scores: pd.DataFrame) -> list[str]:
    """Return sample columns from a feature-score table."""
    metadata = set(FEATURE_SCORE_METADATA_COLUMNS)
    return [column for column in feature_scores.columns if column not in metadata]


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.where(scale > 1e-12, scale, 1.0)


def _gene_counts_from_transcripts(
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    gene_ids: list[str] = []
    rows: list[np.ndarray] = []
    for gene_id, frame in transcript_table.groupby("gene_id", sort=True):
        indices = frame.index.to_numpy()
        gene_ids.append(str(gene_id))
        rows.append(transcript_counts[indices].sum(axis=0))
    matrix = np.vstack(rows) if rows else np.zeros((0, transcript_counts.shape[1]))
    gene_table = pd.DataFrame({"gene_id": gene_ids})
    return matrix, gene_table


def abundance_channel(
    gene_counts: np.ndarray,
    gene_table: pd.DataFrame,
    lib_size: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build standardized log-CPM abundance rows for genes."""
    if "gene_id" not in gene_table.columns:
        raise ValueError("gene_table must contain a 'gene_id' column")
    counts = gene_counts.astype(float)
    if lib_size is None:
        lib_size = counts.sum(axis=0)
    safe_lib_size = np.clip(np.asarray(lib_size, dtype=float), 1e-12, None)
    cpm = counts / safe_lib_size[None, :] * 1e6
    matrix = _zscore_rows(np.log2(cpm + 0.5))
    info = pd.DataFrame(
        {
            "feature_id": [
                f"{gene_id}::abundance" for gene_id in gene_table["gene_id"].astype(str)
            ],
            "gene_id": gene_table["gene_id"].astype(str).tolist(),
            "feature_type": "abundance",
        }
    )
    return matrix, info


def gene_feature_channels(
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    gene_counts: np.ndarray | None = None,
    gene_table: pd.DataFrame | None = None,
    switch_design: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return multiplex gene feature channels as rows by samples.

    The output always includes an abundance channel for each gene. Switch channels
    are included only for genes with at least two retained transcripts.

    ``switch_design`` is forwarded to :func:`gene_switch_coordinates` to regress
    nuisance covariates out of each gene's CLR composition before PC1.
    """
    switch_matrix, switch_info = gene_switch_coordinates(
        transcript_counts, transcript_table, design=switch_design
    )
    if switch_info.empty:
        switch_info = pd.DataFrame(columns=["gene_id", "explained_variance_proxy", "n_transcripts"])
    switch_info = switch_info.copy()
    if not switch_info.empty:
        switch_info["feature_id"] = switch_info["gene_id"].astype(str) + "::switch"
        switch_info["feature_type"] = "switch"
        switch_info = switch_info[["feature_id", "gene_id", "feature_type", "n_transcripts"]]

    if gene_counts is None or gene_table is None:
        gene_counts, gene_table = _gene_counts_from_transcripts(transcript_counts, transcript_table)
    abundance_matrix, abundance_info = abundance_channel(gene_counts, gene_table)

    matrices = [abundance_matrix]
    infos = [abundance_info]
    if switch_matrix.size:
        matrices.append(switch_matrix)
        infos.append(switch_info)

    matrix = np.vstack(matrices) if matrices else np.zeros((0, transcript_counts.shape[1]))
    feature_info = pd.concat(infos, ignore_index=True)
    feature_info = feature_info.sort_values(["gene_id", "feature_type", "feature_id"]).reset_index(
        drop=True
    )

    order = []
    lookup = {fid: idx for idx, fid in enumerate(pd.concat(infos, ignore_index=True)["feature_id"])}
    raw_matrix = matrix
    if len(infos) == 2:
        raw_info = pd.concat(infos, ignore_index=True)
        lookup = {fid: idx for idx, fid in enumerate(raw_info["feature_id"])}
        order = [lookup[fid] for fid in feature_info["feature_id"]]
        matrix = raw_matrix[order]
    return matrix, feature_info


def make_feature_scores(
    matrix: np.ndarray,
    feature_info: pd.DataFrame,
    sample_table: pd.DataFrame,
) -> pd.DataFrame:
    sample_ids = (
        sample_table["sample_id"].astype(str).tolist()
        if "sample_id" in sample_table.columns
        else [str(index) for index in range(len(sample_table))]
    )
    scores = pd.DataFrame(matrix, columns=sample_ids)
    return pd.concat([feature_info.reset_index(drop=True), scores], axis=1)
