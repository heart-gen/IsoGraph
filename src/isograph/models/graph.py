"""Stage-3 graph-aware network model.

Extends the Stage-2 latent model with a graph-Laplacian smoothing step
applied to the residualized switch-coordinate matrix before Factor Analysis.
Setting gamma=0 (no smoothing) exactly recovers the Stage-2 LatentNetworkModel.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from isograph.features.channels import gene_feature_channels, make_feature_scores
from isograph.features.graph import graph_laplacian, laplacian_smooth
from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.models.base import (
    FitArtifacts,
    NetworkModel,
    compute_module_gene_roles,
    compute_trait_associations,
)
from isograph.models.denoise import denoise_features_per_channel
from isograph.models.multiplex import project_feature_similarity_to_gene_graph, select_alpha_abundance
from isograph.workflow.config import GraphModelConfig


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
        feature_ids = feature_info["feature_id"].tolist()
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
            "graph_edge_types_used": list(self.config.edge_types),
            "graph_gamma": self.config.gamma,
            "graph_corr_threshold": self.config.corr_threshold,
            "graph_smoothing_rmse": 0.0,
        }
        edge_rows: list[dict] = []

        if n_features >= 2:
            # Build feature-channel graph from residualized feature coordinates.
            feature_graph = nx.Graph()
            feature_graph.add_nodes_from(feature_ids)
            if "corr" in self.config.edge_types and switch_matrix.shape[1] >= 2:
                corr = np.corrcoef(switch_matrix)
                np.fill_diagonal(corr, 0.0)
                rows, cols = np.where(np.abs(corr) >= self.config.corr_threshold)
                for i, j in zip(rows, cols):
                    if i < j:
                        feature_graph.add_edge(feature_ids[i], feature_ids[j], weight=float(abs(corr[i, j])))
            if "same_gene" in self.config.edge_types:
                for _, frame in feature_info.groupby("gene_id"):
                    ids = frame["feature_id"].tolist()
                    for i, source in enumerate(ids):
                        for target in ids[i + 1 :]:
                            feature_graph.add_edge(source, target, weight=1.0)
            L = graph_laplacian(feature_graph, feature_ids, self.config.normalized_laplacian)

            # Apply Laplacian smoothing; gamma=0 returns switch_matrix unchanged
            smoothed = laplacian_smooth(switch_matrix, L, self.config.gamma)
            smoothing_rmse = float(
                np.sqrt(np.mean((smoothed - switch_matrix) ** 2))
            ) if self.config.gamma > 0 else 0.0

            degrees = dict(feature_graph.degree())
            calibration.update({
                "graph_n_nodes": feature_graph.number_of_nodes(),
                "graph_n_edges": feature_graph.number_of_edges(),
                "graph_mean_degree": float(np.mean(list(degrees.values()))),
                "graph_smoothing_rmse": smoothing_rmse,
            })

            # FA per multiplex channel on the smoothed matrix (same denoising as
            # Stage 2): denoise switch and abundance independently so a
            # high-variance channel cannot swamp the others' component selection.
            denoised_switch, fa_calibration = denoise_features_per_channel(
                smoothed,
                feature_info,
                n_components=self.config.n_components,
                n_components_grid=self.config.n_components_grid,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                n_splits=self.config.n_components_cv_folds,
            )
            calibration.update(fa_calibration)

            partial = self._partial_correlation(denoised_switch)
            resolved_alpha_abundance = self.config.alpha_abundance
            if self.config.alpha_abundance_grid is not None:
                resolved_alpha_abundance = select_alpha_abundance(
                    partial, feature_info, self.config.alpha, self.config.alpha_abundance_grid,
                    alpha_switch=self.config.alpha_switch,
                )
                calibration["selected_alpha_abundance"] = resolved_alpha_abundance
            net_graph, edge_rows = project_feature_similarity_to_gene_graph(
                partial, feature_info, self.config.alpha,
                allow_abundance_abundance=self.config.allow_abundance_abundance,
                alpha_switch=self.config.alpha_switch,
                alpha_abundance=resolved_alpha_abundance,
            )

        module_table = self._module_table(net_graph)
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
