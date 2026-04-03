"""Gene-aware compositional transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def transcript_usage(transcript_counts: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    counts = transcript_counts.astype(float) + pseudocount
    gene_totals = counts.sum(axis=0, keepdims=True)
    return counts / np.clip(gene_totals, 1e-12, None)


def clr_transform(proportions: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    safe = np.clip(proportions, eps, 1.0)
    logs = np.log(safe)
    return logs - logs.mean(axis=0, keepdims=True)


def inverse_clr(clr_values: np.ndarray) -> np.ndarray:
    exp_values = np.exp(clr_values)
    return exp_values / exp_values.sum(axis=0, keepdims=True)


def logit_transform(psi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    safe = np.clip(psi, eps, 1 - eps)
    return logit(safe)


def inverse_logit(values: np.ndarray) -> np.ndarray:
    return expit(values)


def group_transcript_clr(
    transcript_counts: np.ndarray, transcript_table: pd.DataFrame
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    gene_groups = []
    matrices = []
    gene_ids = []
    for gene_id, frame in transcript_table.groupby("gene_id", sort=True):
        indices = frame.index.to_numpy()
        if len(indices) < 2:
            continue
        gene_ids.append(gene_id)
        gene_groups.append(indices)
        matrices.append(clr_transform(transcript_usage(transcript_counts[indices])))
    return np.array(gene_ids, dtype=object), gene_groups, matrices
