"""Stage-3 graph-aware network model.

Extends the Stage-2 latent model with a graph-Laplacian smoothing step
applied to the residualized switch-coordinate matrix before Factor Analysis.
Setting gamma=0 (no smoothing) exactly recovers the Stage-2 LatentNetworkModel.
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
from isograph.features.graph import graph_laplacian, laplacian_smooth
from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.models.base import (
    FitArtifacts,
    NetworkModel,
    compute_module_gene_roles,
    compute_trait_associations,
)
from isograph.models.multiplex import project_feature_similarity_to_gene_graph, select_alpha_abundance
from isograph.workflow.config import GraphModelConfig

_log = logging.getLogger(__name__)


def _cv_select_n_components(
    X: np.ndarray,
    grid: list[int],
    max_iter: int,
    tol: float,
    n_splits: int = 5,
) -> int:
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
class GraphNetworkModel(NetworkModel):
    config: GraphModelConfig

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
        all_matrix, feature_info = gene_feature_channels(
            transcript_counts, transcript_table, gene_counts, gene_table
        )
        cfg = self.config
        if all_matrix.size:
            design = build_design_matrix(sample_table, cfg.residualize_covariates)
            all_matrix = residualize_rows(all_matrix, design)

        use_multiplex = cfg.allow_abundance_abundance or cfg.alpha_abundance_grid is not None
        is_switch = feature_info["feature_type"].astype(str).eq("switch").values
        if use_multiplex:
            compute_matrix, compute_info = all_matrix, feature_info
        else:
            compute_matrix = all_matrix[is_switch]
            compute_info = feature_info[is_switch].reset_index(drop=True)

        n_features = compute_matrix.shape[0]
        feature_ids = compute_info["feature_id"].tolist()
        net_graph = nx.Graph()
        net_graph.add_nodes_from(sorted(feature_info["gene_id"].astype(str).unique()))

        calibration: dict = {
            "mean_log_likelihood": None,
            "reconstruction_rmse": None,
            "mean_noise_variance": None,
            "n_components_used": 0,
            "n_iter": 0,
            "converged": True,
            "graph_n_nodes": n_features,
            "graph_n_edges": 0,
            "graph_mean_degree": 0.0,
            "graph_edge_types_used": list(cfg.edge_types),
            "graph_gamma": cfg.gamma,
            "graph_corr_threshold": cfg.corr_threshold,
            "graph_smoothing_rmse": 0.0,
        }
        edge_rows: list[dict] = []

        if n_features >= 2:
            # Build feature-channel graph from residualized feature coordinates.
            feature_graph = nx.Graph()
            feature_graph.add_nodes_from(feature_ids)
            if "corr" in cfg.edge_types and compute_matrix.shape[1] >= 2:
                corr = np.corrcoef(compute_matrix)
                np.fill_diagonal(corr, 0.0)
                rows, cols = np.where(np.abs(corr) >= cfg.corr_threshold)
                for i, j in zip(rows, cols):
                    if i < j:
                        feature_graph.add_edge(feature_ids[i], feature_ids[j], weight=float(abs(corr[i, j])))
            if "same_gene" in cfg.edge_types:
                for _, frame in compute_info.groupby("gene_id"):
                    ids = frame["feature_id"].tolist()
                    for i, source in enumerate(ids):
                        for target in ids[i + 1 :]:
                            feature_graph.add_edge(source, target, weight=1.0)
            L = graph_laplacian(feature_graph, feature_ids, cfg.normalized_laplacian)

            # Apply Laplacian smoothing; gamma=0 returns compute_matrix unchanged
            smoothed = laplacian_smooth(compute_matrix, L, cfg.gamma)
            smoothing_rmse = float(
                np.sqrt(np.mean((smoothed - compute_matrix) ** 2))
            ) if cfg.gamma > 0 else 0.0

            degrees = dict(feature_graph.degree())
            calibration.update({
                "graph_n_nodes": feature_graph.number_of_nodes(),
                "graph_n_edges": feature_graph.number_of_edges(),
                "graph_mean_degree": float(np.mean(list(degrees.values()))),
                "graph_smoothing_rmse": smoothing_rmse,
            })

            # FA on smoothed matrix (same CV-selection logic as Stage 2)
            X = smoothed.T  # (n_samples, n_features)
            if cfg.n_components_grid:
                k = _cv_select_n_components(
                    X,
                    cfg.n_components_grid,
                    cfg.max_iter,
                    cfg.tol,
                    n_splits=cfg.n_components_cv_folds,
                )
            else:
                k = max(1, min(cfg.n_components, X.shape[1] - 1, X.shape[0] - 1))

            fa = FactorAnalysis(
                n_components=k,
                max_iter=cfg.max_iter,
                tol=cfg.tol,
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
                        cfg.max_iter,
                        n_features,
                        X.shape[0],
                        k,
                    )

            Z = fa.transform(X)
            X_denoised = Z @ fa.components_ + fa.mean_
            denoised_switch = X_denoised.T

            log_ll = float(fa.score(X))
            rmse = float(np.sqrt(np.mean((X - X_denoised) ** 2)))
            calibration.update({
                "mean_log_likelihood": log_ll,
                "reconstruction_rmse": rmse,
                "mean_noise_variance": float(fa.noise_variance_.mean()),
                "n_components_used": k,
                "n_components_selected_by": "cv" if cfg.n_components_grid else "fixed",
                "n_iter": int(fa.n_iter_),
                "converged": converged,
            })

            partial = self._partial_correlation(denoised_switch)
            resolved_alpha_abundance = cfg.alpha_abundance
            if cfg.alpha_abundance_grid is not None:
                resolved_alpha_abundance = select_alpha_abundance(
                    partial, compute_info, cfg.alpha, cfg.alpha_abundance_grid,
                    alpha_switch=cfg.alpha_switch,
                )
                calibration["selected_alpha_abundance"] = resolved_alpha_abundance
            net_graph, edge_rows = project_feature_similarity_to_gene_graph(
                partial, compute_info, cfg.alpha,
                allow_abundance_abundance=cfg.allow_abundance_abundance,
                alpha_switch=cfg.alpha_switch,
                alpha_abundance=resolved_alpha_abundance,
            )

        module_table = self._module_table(net_graph)
        feature_scores = make_feature_scores(all_matrix, feature_info, sample_table)
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
