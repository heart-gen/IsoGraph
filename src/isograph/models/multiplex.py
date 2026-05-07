"""Helpers for projecting feature-channel networks to gene networks."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def giant_component_fraction(graph: nx.Graph, n_genes: int) -> float:
    """Fraction of genes in the largest connected component."""
    if n_genes == 0 or graph.number_of_nodes() == 0:
        return 0.0
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    return len(components[0]) / n_genes if components else 0.0


def select_alpha_abundance(
    similarity: np.ndarray,
    feature_info: pd.DataFrame,
    alpha: float,
    alpha_abundance_grid: list[float],
    alpha_switch: float | None = None,
) -> float:
    """Select the smallest alpha_abundance that does not merge any switch-based modules.

    Builds a baseline gene graph without abundance-abundance edges, identifies its
    multi-gene connected components (the switch-based modules), then sweeps the grid
    ascending and returns the first alpha where no two baseline components are joined
    (directly or transitively through abundance-only genes) by the new edges.
    Falls back to max(grid) if every candidate causes merging or if the baseline has
    no multi-gene components.
    """
    baseline_graph, _ = project_feature_similarity_to_gene_graph(
        similarity, feature_info, alpha,
        allow_abundance_abundance=False,
        alpha_switch=alpha_switch,
    )
    baseline_comps = [
        frozenset(c) for c in nx.connected_components(baseline_graph) if len(c) >= 2
    ]
    if not baseline_comps:
        return max(alpha_abundance_grid)

    # Map each gene to its baseline component index (singletons excluded → None).
    gene_to_baseline: dict[str, int] = {}
    for idx, comp in enumerate(baseline_comps):
        for gene in comp:
            gene_to_baseline[gene] = idx

    for candidate in sorted(alpha_abundance_grid):
        graph, _ = project_feature_similarity_to_gene_graph(
            similarity, feature_info, alpha,
            allow_abundance_abundance=True,
            alpha_abundance=candidate,
            alpha_switch=alpha_switch,
        )
        merging = False
        for new_comp in nx.connected_components(graph):
            # Collect the distinct baseline-component IDs present in this new component.
            baseline_ids = {gene_to_baseline[g] for g in new_comp if g in gene_to_baseline}
            if len(baseline_ids) > 1:
                merging = True
                break
        if not merging:
            return candidate

    return max(alpha_abundance_grid)


def project_feature_similarity_to_gene_graph(
    similarity: np.ndarray,
    feature_info: pd.DataFrame,
    alpha: float,
    allow_abundance_abundance: bool = False,
    alpha_switch: float | None = None,
    alpha_abundance: float | None = None,
) -> tuple[nx.Graph, list[dict[str, object]]]:
    """Aggregate feature-channel similarities into a gene-level graph.

    Switch↔abundance cross-channel edges between dual-channel genes are always
    suppressed (use the dedicated switch-switch edge instead). Abundance↔abundance
    edges between dual-channel genes are included only when allow_abundance_abundance
    is True; use alpha_abundance (or alpha_abundance_grid via select_alpha_abundance)
    to calibrate the per-channel threshold and avoid giant components.
    """
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
            # Block switch↔abundance cross-channel edges between dual-channel genes;
            # abundance↔abundance edges are handled solely by allow_abundance_abundance.
            if (
                source_type != target_type
                and source_gene in genes_with_switch
                and target_gene in genes_with_switch
            ):
                continue
            weight = float(similarity[i, j])
            # Resolve per-channel threshold.
            if alpha_switch is not None and source_type == "switch" and target_type == "switch":
                effective_alpha = alpha_switch
            elif alpha_abundance is not None and source_type == "abundance" and target_type == "abundance":
                effective_alpha = alpha_abundance
            else:
                effective_alpha = alpha
            if not np.isfinite(weight) or abs(weight) < effective_alpha:
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
