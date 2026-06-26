"""Stage-2 probabilistic latent network model.

Uses Factor Analysis as a probabilistic denoiser on the CLR switch-coordinate matrix,
then applies Ledoit-Wolf partial correlation on the FA-reconstructed (denoised) data
to infer the sparse gene network. The FA step introduces calibrated uncertainty estimates
(log-likelihood, reconstruction RMSE, per-gene noise variance) that are absent in the
deterministic Stage-1 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from isograph.features.channels import gene_feature_channels, make_feature_scores
from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.models.base import (
    FitArtifacts,
    NetworkModel,
    compute_module_gene_roles,
    compute_trait_associations,
)
from isograph.models.denoise import denoise_features_per_channel
from isograph.models.multiplex import project_feature_similarity_to_gene_graph, select_alpha_abundance
from isograph.workflow.config import LatentModelConfig


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
            # Probabilistic denoising: FA per multiplex channel (switch / abundance)
            # so a high-variance channel cannot swamp the others' component selection.
            denoised_switch, calibration = denoise_features_per_channel(
                switch_matrix,
                feature_info,
                n_components=self.config.n_components,
                n_components_grid=self.config.n_components_grid,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                n_splits=self.config.n_components_cv_folds,
            )

            # Gene network: strongest feature-channel evidence per gene pair
            partial = self._partial_correlation(denoised_switch)
            resolved_alpha_abundance = self.config.alpha_abundance
            if self.config.alpha_abundance_grid is not None:
                resolved_alpha_abundance = select_alpha_abundance(
                    partial, feature_info, self.config.alpha, self.config.alpha_abundance_grid,
                    alpha_switch=self.config.alpha_switch,
                )
                calibration["selected_alpha_abundance"] = resolved_alpha_abundance
            graph, edge_rows = project_feature_similarity_to_gene_graph(
                partial, feature_info, self.config.alpha,
                allow_abundance_abundance=self.config.allow_abundance_abundance,
                alpha_switch=self.config.alpha_switch,
                alpha_abundance=resolved_alpha_abundance,
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
