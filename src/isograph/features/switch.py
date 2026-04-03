"""Gene switch coordinate construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from isograph.features.composition import group_transcript_clr
from isograph.utils import stable_sign


def gene_switch_coordinates(
    transcript_counts: np.ndarray, transcript_table: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    gene_ids, _, matrices = group_transcript_clr(transcript_counts, transcript_table)
    coordinates = []
    loading_rows: list[dict[str, object]] = []
    for gene_id, clr_values in zip(gene_ids, matrices, strict=False):
        centered = clr_values.T - clr_values.T.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        loadings = vh[0]
        scores = centered @ loadings
        scores, loadings = stable_sign(scores, loadings)
        coordinates.append(scores)
        loading_rows.append(
            {
                "gene_id": gene_id,
                "explained_variance_proxy": float(np.var(scores)),
                "n_transcripts": int(clr_values.shape[0]),
            }
        )
    matrix = np.vstack(coordinates) if coordinates else np.zeros((0, transcript_counts.shape[1]))
    return matrix, pd.DataFrame(loading_rows)
