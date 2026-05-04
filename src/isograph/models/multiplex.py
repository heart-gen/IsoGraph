"""Helpers for projecting feature-channel networks to gene networks."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def project_feature_similarity_to_gene_graph(
    similarity: np.ndarray,
    feature_info: pd.DataFrame,
    alpha: float,
    allow_abundance_abundance: bool = False,
) -> tuple[nx.Graph, list[dict[str, object]]]:
    """Aggregate feature-channel similarities into a gene-level graph."""
    gene_ids = sorted(feature_info["gene_id"].astype(str).unique())
    genes_with_switch = set(
        feature_info.loc[feature_info["feature_type"].astype(str) == "switch", "gene_id"].astype(str)
    )
    graph = nx.Graph()
    graph.add_nodes_from(gene_ids)
    best: dict[tuple[str, str], dict[str, object]] = {}

    for i, source_feature in feature_info.iterrows():
        source_gene = str(source_feature["gene_id"])
        for j in range(i + 1, len(feature_info)):
            target_feature = feature_info.iloc[j]
            target_gene = str(target_feature["gene_id"])
            if source_gene == target_gene:
                continue
            source_type = str(source_feature["feature_type"])
            target_type = str(target_feature["feature_type"])
            if (
                not allow_abundance_abundance
                and source_type == "abundance"
                and target_type == "abundance"
            ):
                continue
            if (
                "abundance" in {source_type, target_type}
                and source_gene in genes_with_switch
                and target_gene in genes_with_switch
            ):
                continue
            weight = float(similarity[i, j])
            if not np.isfinite(weight) or abs(weight) < alpha:
                continue
            source, target = sorted([source_gene, target_gene])
            key = (source, target)
            candidate = {
                "source": source,
                "target": target,
                "weight": weight,
                "source_feature_id": source_feature["feature_id"],
                "target_feature_id": target_feature["feature_id"],
                "source_feature_type": source_type,
                "target_feature_type": target_type,
            }
            current = best.get(key)
            if current is None or abs(weight) > abs(float(current["weight"])):
                best[key] = candidate

    edge_rows = sorted(best.values(), key=lambda row: (row["source"], row["target"]))
    for row in edge_rows:
        graph.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    return graph, edge_rows
