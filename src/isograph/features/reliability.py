"""Per-gene switch-reliability scoring for degradation robustness.

3' RNA degradation (unavoidable in postmortem tissue) perturbs within-gene
isoform composition, rotating a gene's switch coordinate (PC1 of CLR composition)
toward the degradation axis. For such a gene the switch channel is unreliable and
the gene should fall back to its degradation-robust abundance channel.

This module quantifies, per gene, how much of the compositional variation is
aligned with a sample-level degradation metric (RIN, TIN, 3'/5' bias, exonic
coverage slope, ...), and converts that into a reliability weight in [floor, 1].
The weight is consumed by the multiplex graph projection to downweight
switch-switch edges from degradation-dominated genes (abundance fallback).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from isograph.features.composition import group_transcript_clr


def degradation_direction(
    sample_table: pd.DataFrame, covariate: str
) -> np.ndarray | None:
    """Unit-norm, mean-centered sample vector for a degradation covariate.

    Returns None when the covariate is absent, non-numeric, or constant (no
    degradation axis to project onto).
    """
    if covariate not in sample_table.columns:
        return None
    series = sample_table[covariate]
    if not pd.api.types.is_numeric_dtype(series):
        return None
    values = series.to_numpy(dtype=float)
    if np.isnan(values).any():
        values = np.nan_to_num(values, nan=float(np.nanmean(values)))
    centered = values - values.mean()
    norm = np.linalg.norm(centered)
    if norm < 1e-12:
        return None
    return centered / norm


def gene_degradation_sensitivity(
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    direction: np.ndarray,
) -> dict[str, float]:
    """Per-gene fraction of CLR composition variance aligned with ``direction``.

    For each multi-transcript gene, build the centered CLR composition matrix C
    (n_transcripts x n_samples) and compute the share of its total (Frobenius)
    variance captured by the rank-1 ``direction`` axis:

        sensitivity = ||C @ d_hat||^2 / ||C||_F^2  in [0, 1]

    where ``d_hat`` is the unit degradation vector. 1.0 means the gene's
    composition varies almost entirely along the degradation axis (switch signal
    untrustworthy); 0.0 means the composition is orthogonal to degradation.
    """
    gene_ids, _, matrices = group_transcript_clr(transcript_counts, transcript_table)
    d = direction
    sensitivity: dict[str, float] = {}
    for gene_id, clr in zip(gene_ids, matrices, strict=False):
        centered = clr - clr.mean(axis=1, keepdims=True)
        total = float(np.sum(centered * centered))
        if total < 1e-12:
            sensitivity[str(gene_id)] = 0.0
            continue
        aligned = centered @ d  # (n_transcripts,)
        sensitivity[str(gene_id)] = float(np.sum(aligned * aligned) / total)
    return sensitivity


def gene_switch_reliability(
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    direction: np.ndarray,
    floor: float = 0.0,
    power: float = 1.0,
) -> dict[str, float]:
    """Per-gene switch reliability weight in ``[floor, 1]``.

        reliability = clip((1 - sensitivity) ** power, floor, 1)

    ``power`` > 1 sharpens the penalty on degradation-aligned genes; ``floor``
    keeps a minimum switch contribution so genes are never fully zeroed.
    """
    sensitivity = gene_degradation_sensitivity(transcript_counts, transcript_table, direction)
    reliability: dict[str, float] = {}
    for gene_id, s in sensitivity.items():
        r = (1.0 - s) ** power
        reliability[gene_id] = float(min(1.0, max(floor, r)))
    return reliability
