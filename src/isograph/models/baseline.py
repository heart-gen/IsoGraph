"""Deterministic stage-1 baseline network model."""

from __future__ import annotations

from dataclasses import dataclass

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
from isograph.models.multiplex import (
    project_feature_similarity_to_gene_graph,
    select_alpha_abundance,
)
from isograph.workflow.config import BaselineModelConfig


@dataclass
class BaselineNetworkModel(NetworkModel):
    config: BaselineModelConfig

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
        self, module_table: pd.DataFrame, feature_scores: pd.DataFrame, sample_table: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return compute_trait_associations(
            module_table, feature_scores, sample_table, self.config.trait_columns
        )

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
        partial = self._partial_correlation(switch_matrix)
        cfg = self.config
        resolved_alpha_abundance = cfg.alpha_abundance
        if cfg.alpha_abundance_grid is not None:
            resolved_alpha_abundance = select_alpha_abundance(
                partial,
                feature_info,
                cfg.alpha,
                cfg.alpha_abundance_grid,
                alpha_switch=cfg.alpha_switch,
            )
        graph, edge_rows = project_feature_similarity_to_gene_graph(
            partial,
            feature_info,
            cfg.alpha,
            allow_abundance_abundance=cfg.allow_abundance_abundance,
            alpha_switch=cfg.alpha_switch,
            alpha_abundance=resolved_alpha_abundance,
        )
        module_table = self._module_table(graph)
        feature_scores = make_feature_scores(switch_matrix, feature_info, sample_table)
        trait_table, eigengene_table = self._trait_associations(
            module_table, feature_scores, sample_table
        )
        module_gene_roles = compute_module_gene_roles(module_table, feature_scores, sample_table)
        calibration: dict | None = None
        if cfg.allow_abundance_abundance or cfg.alpha_abundance_grid is not None:
            calibration = {"selected_alpha_abundance": resolved_alpha_abundance}
        return FitArtifacts(
            module_table=module_table,
            edge_table=pd.DataFrame(edge_rows),
            trait_table=trait_table,
            feature_scores=feature_scores,
            calibration=calibration,
            eigengene_table=eigengene_table,
            module_gene_roles=module_gene_roles,
        )
