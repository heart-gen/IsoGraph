"""Stage-2 probabilistic latent network model.

Uses Factor Analysis as a probabilistic denoiser on the CLR switch-coordinate matrix,
then applies Ledoit-Wolf partial correlation on the FA-reconstructed (denoised) data
to infer the sparse gene network. The FA step introduces calibrated uncertainty estimates
(log-likelihood, reconstruction RMSE, per-gene noise variance) that are absent in the
deterministic Stage-1 baseline.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from isograph.features.channels import gene_feature_channels, make_feature_scores
from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.models.base import (
    FitArtifacts,
    NetworkModel,
    compute_module_gene_roles,
    compute_trait_associations,
)
from isograph.models.multiplex import project_feature_similarity_to_gene_graph
from isograph.workflow.config import LatentModelConfig

_log = logging.getLogger(__name__)


def _cv_select_n_components(
    X: np.ndarray,
    grid: list[int],
    max_iter: int,
    tol: float,
    n_splits: int = 5,
) -> int:
    """Select n_components from *grid* by cross-validated held-out log-likelihood.

    Fits FA on (n_splits-1) folds and scores on the held-out fold. Picks the
    smallest k whose mean CV log-likelihood is within numerical noise of the
    maximum — this guards against selecting a larger k when gains are marginal.
    Held-out scoring penalises factors that fit training noise and do not
    generalise, giving a reliable elbow even on low-noise synthetic data.
    """
    n_samples, n_genes = X.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    best_k, best_score = grid[0], -np.inf
    for k in grid:
        k_eff = max(1, min(k, n_genes - 1, n_samples - 1))
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


@dataclass
class LatentNetworkModel(NetworkModel):
    config: LatentModelConfig

    def _partial_correlation(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros((0, 0))
        sample_matrix = matrix.T
        covariance = LedoitWolf().fit(sample_matrix).covariance_
        precision = np.linalg.pinv(covariance)
        diagonal = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
        partial = -precision / np.outer(diagonal, diagonal)
        np.fill_diagonal(partial, 0.0)
        return partial

    def _module_table(self, graph: nx.Graph) -> pd.DataFrame:
        rows = []
        for module_index, nodes in enumerate(
            sorted(nx.connected_components(graph), key=len, reverse=True)
        ):
            if len(nodes) < self.config.min_module_size:
                continue
            for gene_id in sorted(nodes):
                rows.append({"gene_id": gene_id, "module_id": f"M{module_index:03d}"})
        return pd.DataFrame(rows)

    def _trait_associations(
        self,
        module_table: pd.DataFrame,
        feature_scores: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return compute_trait_associations(module_table, feature_scores, sample_table, self.config.trait_columns)

    def fit(
        self,
        transcript_counts: np.ndarray,
        transcript_table: pd.DataFrame,
        sample_table: pd.DataFrame,
        gene_counts: np.ndarray | None = None,
        gene_table: pd.DataFrame | None = None,
    ) -> FitArtifacts:
        switch_matrix, feature_info = gene_feature_channels(
            transcript_counts, transcript_table, gene_counts, gene_table
        )
        if switch_matrix.size:
            design = build_design_matrix(sample_table, self.config.residualize_covariates)
            switch_matrix = residualize_rows(switch_matrix, design)

        n_features = switch_matrix.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(sorted(feature_info["gene_id"].astype(str).unique()))

        calibration: dict = {
            "mean_log_likelihood": None,
            "reconstruction_rmse": None,
            "mean_noise_variance": None,
            "n_components_used": 0,
            "n_iter": 0,
            "converged": True,
        }
        edge_rows: list[dict] = []

        if n_features >= 2:
            # X is (n_samples, n_features) — standard orientation for sklearn
            X = switch_matrix.T
            if self.config.n_components_grid:
                k = _cv_select_n_components(
                    X,
                    self.config.n_components_grid,
                    self.config.max_iter,
                    self.config.tol,
                    n_splits=self.config.n_components_cv_folds,
                )
            else:
                k = max(1, min(self.config.n_components, X.shape[1] - 1, X.shape[0] - 1))
            fa = FactorAnalysis(
                n_components=k,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                random_state=0,
            )
            converged = True
            with warnings.catch_warnings(record=True) as _caught:
                warnings.simplefilter("always")
                fa.fit(X)
                if any(issubclass(w.category, ConvergenceWarning) for w in _caught):
                    converged = False
                    _log.warning(
                        "FactorAnalysis did not converge in %d iterations "
                        "(n_features=%d, n_samples=%d, n_components=%d). "
                        "Consider increasing max_iter or tol.",
                        self.config.max_iter,
                        n_features,
                        X.shape[0],
                        k,
                    )

            # Probabilistic denoising: posterior mean reconstruction
            Z = fa.transform(X)                          # (n_samples, k)
            X_denoised = Z @ fa.components_ + fa.mean_  # (n_samples, n_genes)
            denoised_switch = X_denoised.T               # (n_genes, n_samples)

            # Calibration metrics from the FA model
            log_ll = float(fa.score(X))
            rmse = float(np.sqrt(np.mean((X - X_denoised) ** 2)))
            calibration = {
                "mean_log_likelihood": log_ll,
                "reconstruction_rmse": rmse,
                "mean_noise_variance": float(fa.noise_variance_.mean()),
                "n_components_used": k,
                "n_components_selected_by": "cv" if self.config.n_components_grid else "fixed",
                "n_iter": int(fa.n_iter_),
                "converged": converged,
            }

            # Gene network: strongest feature-channel evidence per gene pair
            partial = self._partial_correlation(denoised_switch)
            graph, edge_rows = project_feature_similarity_to_gene_graph(
                partial, feature_info, self.config.alpha
            )

        module_table = self._module_table(graph)
        feature_scores = make_feature_scores(switch_matrix, feature_info, sample_table)
        trait_table, eigengene_table = self._trait_associations(module_table, feature_scores, sample_table)
        module_gene_roles = compute_module_gene_roles(module_table, feature_scores, sample_table)

        return FitArtifacts(
            module_table=module_table,
            edge_table=pd.DataFrame(edge_rows),
            trait_table=trait_table,
            feature_scores=feature_scores,
            calibration=calibration,
            eigengene_table=eigengene_table,
            module_gene_roles=module_gene_roles,
        )
