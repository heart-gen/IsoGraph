"""Alpha selection via stability selection for real data without ground truth.

Stability selection (Meinshausen & Bühlmann, 2010) estimates the probability
that each edge appears when the model is refit on random subsamples of the data.
Edges that appear consistently across subsamples are considered stable. The
method provides a principled way to choose the partial-correlation threshold
(alpha) when module ground truth is unavailable.

Algorithm
---------
For each alpha in a user-supplied grid:
  For each of n_iterations bootstrap rounds:
    Draw a random subsample of `subsample_fraction` of the samples.
    Fit the model on the subsampled data.
    Record which gene pairs appear as edges.
  Compute the stability score for each pair: fraction of rounds with an edge.
Report the number of stable edges (score ≥ stability_threshold) per alpha.

Recommended alpha: the coarsest (largest) alpha for which the stable-edge count
still meets a minimum desired edge count, or the elbow of the stable-edge-count
curve. A lower alpha yields denser networks; a higher alpha yields sparser ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from isograph.models.base import NetworkModel


@dataclass
class StabilityResult:
    """Results from a stability selection run."""

    alpha_grid: list[float]
    stable_edge_counts: list[int]
    recommended_alpha: float
    # Per-alpha edge stability scores: dict[alpha -> dict[frozenset(pair) -> float]]
    edge_stability: dict[float, dict[frozenset, float]]

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"alpha": self.alpha_grid, "stable_edge_count": self.stable_edge_counts}
        )


def stability_selection(
    model: "NetworkModel",
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    sample_table: pd.DataFrame,
    alpha_grid: list[float],
    n_iterations: int = 50,
    subsample_fraction: float = 0.8,
    stability_threshold: float = 0.6,
    seed: int = 0,
) -> StabilityResult:
    """Estimate edge stability across subsamples for each alpha in ``alpha_grid``.

    Parameters
    ----------
    model:
        A fitted (or unfitted) NetworkModel instance. The model's config will
        be temporarily patched with each alpha from the grid.
    transcript_counts:
        Shape (n_transcripts, n_samples). Raw count matrix.
    transcript_table:
        Rows describe each transcript. Must contain ``gene_id`` and
        ``transcript_id`` columns.
    sample_table:
        One row per sample. Must be aligned with columns of transcript_counts.
    alpha_grid:
        Sorted (ascending) list of partial-correlation threshold values to test.
    n_iterations:
        Number of subsampling rounds per alpha. Higher values give more stable
        estimates; 50 is sufficient for most datasets.
    subsample_fraction:
        Fraction of samples to draw per round (without replacement).
    stability_threshold:
        Minimum fraction of rounds in which a gene pair must appear as an edge
        to be counted as a stable edge.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    StabilityResult
        Contains stable-edge counts per alpha and a recommended alpha.
    """
    import dataclasses

    rng = np.random.default_rng(seed)
    n_samples = transcript_counts.shape[1]
    subsample_size = max(2, int(round(n_samples * subsample_fraction)))

    # If the model uses CV component selection, pre-select k once on the full
    # dataset and fix it for all subsample iterations. Running CV inside every
    # subsample fit would be thousands of redundant FA fits and would also pick
    # different k on different subsamples, making stability scores incomparable.
    if getattr(model.config, "n_components_grid", None):
        probe = model.fit(
            transcript_counts=transcript_counts,
            transcript_table=transcript_table,
            sample_table=sample_table,
        )
        k_fixed = probe.calibration["n_components_used"]
        model = type(model)(
            dataclasses.replace(model.config, n_components=k_fixed, n_components_grid=None)
        )

    edge_stability: dict[float, dict[frozenset, float]] = {}

    for alpha in alpha_grid:
        patched_config = dataclasses.replace(model.config, alpha=alpha)
        patched_model = type(model)(patched_config)

        edge_counts: dict[frozenset, int] = {}

        for _ in range(n_iterations):
            idx = rng.choice(n_samples, size=subsample_size, replace=False)
            idx_sorted = np.sort(idx)

            sub_counts = transcript_counts[:, idx_sorted]
            sub_samples = sample_table.iloc[idx_sorted].reset_index(drop=True)

            try:
                artifacts = patched_model.fit(
                    transcript_counts=sub_counts,
                    transcript_table=transcript_table,
                    sample_table=sub_samples,
                )
            except Exception:
                continue

            for row in artifacts.edge_table.itertuples(index=False):
                pair = frozenset([row.source, row.target])
                edge_counts[pair] = edge_counts.get(pair, 0) + 1

        stability: dict[frozenset, float] = {
            pair: count / n_iterations for pair, count in edge_counts.items()
        }
        edge_stability[alpha] = stability

    stable_edge_counts = [
        sum(1 for s in edge_stability[a].values() if s >= stability_threshold)
        for a in alpha_grid
    ]

    # Recommended alpha: largest alpha with at least 1 stable edge.
    recommended_alpha = alpha_grid[-1]
    for a, count in zip(alpha_grid, stable_edge_counts):
        if count > 0:
            recommended_alpha = a

    return StabilityResult(
        alpha_grid=alpha_grid,
        stable_edge_counts=stable_edge_counts,
        recommended_alpha=recommended_alpha,
        edge_stability=edge_stability,
    )
