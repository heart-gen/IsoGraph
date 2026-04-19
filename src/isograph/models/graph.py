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
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from isograph.features.graph import build_gene_graph, graph_laplacian, laplacian_smooth
from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.features.switch import gene_switch_coordinates
from isograph.models.base import FitArtifacts, NetworkModel
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
    ) -> pd.DataFrame:
        rows = []
        if module_table.empty:
            return pd.DataFrame(columns=["module_id", "trait", "effect", "pvalue"])
        for module_id in sorted(module_table["module_id"].unique()):
            genes = module_table.loc[module_table["module_id"] == module_id, "gene_id"]
            subset = feature_scores.loc[feature_scores["gene_id"].isin(genes)]
            eigengene = subset.drop(columns=["gene_id"]).to_numpy(dtype=float).mean(axis=0)
            if "Age" in sample_table.columns:
                effect, pvalue = stats.pearsonr(
                    eigengene, sample_table["Age"].to_numpy(dtype=float)
                )
                rows.append({"module_id": module_id, "trait": "Age", "effect": effect, "pvalue": pvalue})
            if "Dx" in sample_table.columns:
                dx = (sample_table["Dx"] == "SCZD").astype(float).to_numpy()
                ttest = stats.ttest_ind(eigengene[dx == 0], eigengene[dx == 1], equal_var=False)
                effect = float(eigengene[dx == 1].mean() - eigengene[dx == 0].mean())
                rows.append({"module_id": module_id, "trait": "Dx", "effect": effect, "pvalue": float(ttest.pvalue)})
        return pd.DataFrame(rows)

    def fit(
        self,
        transcript_counts: np.ndarray,
        transcript_table: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> FitArtifacts:
        switch_matrix, feature_info = gene_switch_coordinates(transcript_counts, transcript_table)
        if switch_matrix.size:
            design = build_design_matrix(sample_table, self.config.residualize_covariates)
            switch_matrix = residualize_rows(switch_matrix, design)

        n_genes = switch_matrix.shape[0]
        gene_ids = feature_info["gene_id"].tolist()
        net_graph = nx.Graph()
        net_graph.add_nodes_from(gene_ids)

        calibration: dict = {
            "mean_log_likelihood": None,
            "reconstruction_rmse": None,
            "mean_noise_variance": None,
            "n_components_used": 0,
            "n_iter": 0,
            "converged": True,
            "graph_n_nodes": n_genes,
            "graph_n_edges": 0,
            "graph_mean_degree": 0.0,
            "graph_edge_types_used": list(self.config.edge_types),
            "graph_gamma": self.config.gamma,
            "graph_corr_threshold": self.config.corr_threshold,
            "graph_smoothing_rmse": 0.0,
        }
        edge_rows: list[dict] = []

        if n_genes >= 2:
            # Build gene graph from residualized switch coordinates
            gene_graph = build_gene_graph(
                switch_matrix,
                gene_ids,
                transcript_table,
                self.config.edge_types,
                self.config.corr_threshold,
            )
            L = graph_laplacian(gene_graph, gene_ids, self.config.normalized_laplacian)

            # Apply Laplacian smoothing; gamma=0 returns switch_matrix unchanged
            smoothed = laplacian_smooth(switch_matrix, L, self.config.gamma)
            smoothing_rmse = float(
                np.sqrt(np.mean((smoothed - switch_matrix) ** 2))
            ) if self.config.gamma > 0 else 0.0

            degrees = dict(gene_graph.degree())
            calibration.update({
                "graph_n_nodes": gene_graph.number_of_nodes(),
                "graph_n_edges": gene_graph.number_of_edges(),
                "graph_mean_degree": float(np.mean(list(degrees.values()))),
                "graph_smoothing_rmse": smoothing_rmse,
            })

            # FA on smoothed matrix (same CV-selection logic as Stage 2)
            X = smoothed.T  # (n_samples, n_genes)
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
                        "(n_genes=%d, n_samples=%d, n_components=%d). "
                        "Consider increasing max_iter or tol.",
                        self.config.max_iter,
                        n_genes,
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
                "n_components_selected_by": "cv" if self.config.n_components_grid else "fixed",
                "n_iter": int(fa.n_iter_),
                "converged": converged,
            })

            partial = self._partial_correlation(denoised_switch)
            for i, source in enumerate(gene_ids):
                for j in range(i + 1, n_genes):
                    weight = float(partial[i, j])
                    if abs(weight) < self.config.alpha:
                        continue
                    target = gene_ids[j]
                    net_graph.add_edge(source, target, weight=weight)
                    edge_rows.append({"source": source, "target": target, "weight": weight})

        module_table = self._module_table(net_graph)
        feature_scores = pd.DataFrame(switch_matrix, index=feature_info["gene_id"])
        feature_scores = feature_scores.reset_index().rename(columns={"index": "gene_id"})
        trait_table = self._trait_associations(module_table, feature_scores, sample_table)

        return FitArtifacts(
            module_table=module_table,
            edge_table=pd.DataFrame(edge_rows),
            trait_table=trait_table,
            feature_scores=feature_scores,
            calibration=calibration,
        )
