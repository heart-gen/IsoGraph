"""Deterministic stage-1 baseline network model."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.features.switch import gene_switch_coordinates
from isograph.models.base import FitArtifacts, NetworkModel
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


    def _module_table(self, graph: nx.Graph) -> pd.DataFrame:
        rows = []
        for module_index, nodes in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True)):
            if len(nodes) < self.config.min_module_size:
                continue
            for gene_id in sorted(nodes):
                rows.append({"gene_id": gene_id, "module_id": f"M{module_index:03d}"})
        return pd.DataFrame(rows)

    def _trait_associations(
        self, module_table: pd.DataFrame, feature_scores: pd.DataFrame, sample_table: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sample_ids = sample_table["sample_id"].tolist() if "sample_id" in sample_table.columns else list(range(len(sample_table)))
        if module_table.empty:
            return (
                pd.DataFrame(columns=["module_id", "trait", "effect", "pvalue"]),
                pd.DataFrame(columns=["module_id"] + sample_ids),
            )
        rows = []
        eigengene_rows: dict[str, list] = {}
        module_ids = sorted(module_table["module_id"].unique())
        for module_id in module_ids:
            genes = module_table.loc[module_table["module_id"] == module_id, "gene_id"]
            subset = feature_scores.loc[feature_scores["gene_id"].isin(genes)]
            eigengene = subset.drop(columns=["gene_id"]).to_numpy(dtype=float).mean(axis=0)
            eigengene_rows[module_id] = eigengene.tolist()
            if "Age" in sample_table.columns:
                effect, pvalue = stats.pearsonr(eigengene, sample_table["Age"].to_numpy(dtype=float))
                rows.append({"module_id": module_id, "trait": "Age", "effect": effect, "pvalue": pvalue})
            if "Dx" in sample_table.columns:
                dx = (sample_table["Dx"] == "SCZD").astype(float).to_numpy()
                model = stats.ttest_ind(eigengene[dx == 0], eigengene[dx == 1], equal_var=False)
                effect = float(eigengene[dx == 1].mean() - eigengene[dx == 0].mean())
                rows.append({"module_id": module_id, "trait": "Dx", "effect": effect, "pvalue": float(model.pvalue)})
        eigengene_table = pd.DataFrame(eigengene_rows, index=sample_ids).T.reset_index().rename(columns={"index": "module_id"})
        return pd.DataFrame(rows), eigengene_table

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
        graph = nx.Graph()
        graph.add_nodes_from(feature_info["gene_id"].tolist())
        partial = self._partial_correlation(switch_matrix)
        edge_rows = []
        for i, source in enumerate(feature_info["gene_id"]):
            for j in range(i + 1, len(feature_info)):
                weight = float(partial[i, j])
                if abs(weight) < self.config.alpha:
                    continue
                target = feature_info.iloc[j]["gene_id"]
                graph.add_edge(source, target, weight=weight)
                edge_rows.append({"source": source, "target": target, "weight": weight})
        module_table = self._module_table(graph)
        feature_scores = pd.DataFrame(switch_matrix, index=feature_info["gene_id"])
        feature_scores = feature_scores.reset_index().rename(columns={"index": "gene_id"})
        trait_table, eigengene_table = self._trait_associations(module_table, feature_scores, sample_table)
        return FitArtifacts(
            module_table=module_table,
            edge_table=pd.DataFrame(edge_rows),
            trait_table=trait_table,
            feature_scores=feature_scores,
            eigengene_table=eigengene_table,
        )
