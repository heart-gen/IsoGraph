"""Factor-analysis denoising of multiplex feature channels.

Stage 2 (latent) and Stage 3 (graph) denoise the residualized feature-coordinate
matrix with Factor Analysis before estimating the gene network. The matrix mixes
heterogeneous channels -- a switch coordinate and an abundance coordinate per gene
(see :func:`isograph.features.channels.gene_feature_channels`). A *single joint* FA
over both channels lets the higher-variance channel dominate the cross-validated
component selection: the abundance channel is smooth and low-rank, so CV picks a
small ``k`` that captures abundance structure and under-denoises the switch signal,
collapsing module recovery below even the deterministic Stage-1 baseline.

Denoising each channel independently keeps the per-channel latent dimensionality
honest: the switch channel gets the ``k`` its own held-out likelihood supports, and
the abundance channel gets its own. Reassembled in the original feature-row order,
the result is a drop-in replacement for the joint-FA output downstream.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

_log = logging.getLogger(__name__)


def cv_select_n_components(
    X: np.ndarray,
    grid: list[int],
    max_iter: int,
    tol: float,
    n_splits: int = 5,
) -> int:
    """Select ``n_components`` from *grid* by cross-validated held-out log-likelihood.

    Fits FA on (n_splits-1) folds and scores on the held-out fold, picking the ``k``
    with the highest mean CV log-likelihood. Held-out scoring penalises factors that
    fit training noise and do not generalise, giving a reliable elbow even on
    low-noise synthetic data.
    """
    n_samples, n_features = X.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    best_k, best_score = grid[0], -np.inf
    for k in grid:
        k_eff = max(1, min(k, n_features - 1, n_samples - 1))
        fold_scores = []
        for train_idx, test_idx in kf.split(X):
            fa = FactorAnalysis(n_components=k_eff, max_iter=max_iter, tol=tol, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fa.fit(X[train_idx])
            fold_scores.append(float(fa.score(X[test_idx])))
        score = float(np.mean(fold_scores))
        _log.debug("CV sweep k=%d  cv_ll=%.4f", k_eff, score)
        if score > best_score:
            best_score, best_k = score, k_eff
    _log.info("CV selected n_components=%d (cv_ll=%.4f)", best_k, best_score)
    return best_k


def _denoise_channel(
    sub_matrix: np.ndarray,
    *,
    n_components: int,
    n_components_grid: list[int] | None,
    max_iter: int,
    tol: float,
    n_splits: int,
) -> tuple[np.ndarray, dict]:
    """FA-denoise one channel's (n_chan_features, n_samples) sub-matrix.

    Channels with fewer than two features (FA is degenerate) are returned
    unchanged with ``k = 0`` recorded.
    """
    n_chan_features = sub_matrix.shape[0]
    if n_chan_features < 2:
        return sub_matrix, {
            "n_components_used": 0,
            "mean_log_likelihood": None,
            "reconstruction_rmse": 0.0,
            "mean_noise_variance": None,
            "n_iter": 0,
            "converged": True,
        }

    X = sub_matrix.T  # (n_samples, n_chan_features) — sklearn orientation
    if n_components_grid:
        k = cv_select_n_components(X, n_components_grid, max_iter, tol, n_splits=n_splits)
    else:
        k = max(1, min(n_components, X.shape[1] - 1, X.shape[0] - 1))

    fa = FactorAnalysis(n_components=k, max_iter=max_iter, tol=tol, random_state=0)
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fa.fit(X)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False
            _log.warning(
                "FactorAnalysis did not converge in %d iterations "
                "(n_features=%d, n_samples=%d, n_components=%d). "
                "Consider increasing max_iter or tol.",
                max_iter,
                n_chan_features,
                X.shape[0],
                k,
            )

    Z = fa.transform(X)
    X_denoised = Z @ fa.components_ + fa.mean_
    stats = {
        "n_components_used": k,
        "mean_log_likelihood": float(fa.score(X)),
        "reconstruction_rmse": float(np.sqrt(np.mean((X - X_denoised) ** 2))),
        "mean_noise_variance": float(fa.noise_variance_.mean()),
        "n_iter": int(fa.n_iter_),
        "converged": converged,
    }
    return X_denoised.T, stats


def denoise_features_per_channel(
    matrix: np.ndarray,
    feature_info: pd.DataFrame,
    *,
    n_components: int,
    n_components_grid: list[int] | None,
    max_iter: int,
    tol: float,
    n_splits: int = 5,
) -> tuple[np.ndarray, dict]:
    """Denoise a multiplex feature matrix one channel at a time.

    ``matrix`` is ``(n_features, n_samples)`` with rows aligned to ``feature_info``
    (whose ``feature_type`` column names each row's channel). Each channel is
    FA-denoised independently and reassembled in the original row order.

    The returned calibration dict mirrors the joint-FA contract used by the model
    fits: ``n_components_used`` reports the *switch* channel's ``k`` (the primary
    biological signal) when present, else the first denoised channel's ``k``. The
    full per-channel breakdown is under ``n_components_per_channel``. The scalar
    ``reconstruction_rmse`` / ``mean_noise_variance`` are aggregated over the whole
    reassembled matrix; ``converged`` is the conjunction over channels.
    """
    feature_types = feature_info["feature_type"].to_numpy()
    denoised = matrix.copy()

    per_channel: dict[str, dict] = {}
    # Deterministic, switch-first ordering so n_components_used prefers the switch k.
    channels = sorted(pd.unique(feature_types), key=lambda c: (c != "switch", str(c)))
    for channel in channels:
        mask = feature_types == channel
        sub_denoised, stats = _denoise_channel(
            matrix[mask],
            n_components=n_components,
            n_components_grid=n_components_grid,
            max_iter=max_iter,
            tol=tol,
            n_splits=n_splits,
        )
        denoised[mask] = sub_denoised
        per_channel[str(channel)] = stats

    primary = "switch" if "switch" in per_channel else (channels[0] if channels else None)
    primary_stats = per_channel.get(str(primary), {}) if primary is not None else {}

    total_rmse = float(np.sqrt(np.mean((matrix - denoised) ** 2))) if matrix.size else 0.0
    noise_vars = [
        s["mean_noise_variance"]
        for s in per_channel.values()
        if s["mean_noise_variance"] is not None
    ]
    log_lls = [
        s["mean_log_likelihood"]
        for s in per_channel.values()
        if s["mean_log_likelihood"] is not None
    ]
    calibration = {
        "mean_log_likelihood": float(np.mean(log_lls)) if log_lls else None,
        "reconstruction_rmse": total_rmse,
        "mean_noise_variance": float(np.mean(noise_vars)) if noise_vars else None,
        "n_components_used": int(primary_stats.get("n_components_used", 0)),
        "n_components_per_channel": {
            ch: int(s["n_components_used"]) for ch, s in per_channel.items()
        },
        "n_components_selected_by": "cv" if n_components_grid else "fixed",
        "n_iter": max((s["n_iter"] for s in per_channel.values()), default=0),
        "converged": all(s["converged"] for s in per_channel.values()),
    }
    return denoised, calibration
