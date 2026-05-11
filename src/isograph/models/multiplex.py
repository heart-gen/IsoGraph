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

    # Pre-extract arrays once to avoid per-row pandas overhead inside the loop.
    _gene_ids = feature_info["gene_id"].astype(str).to_numpy()
    _feat_types = feature_info["feature_type"].astype(str).to_numpy()
    _feat_ids = feature_info["feature_id"].astype(str).to_numpy()

    # Minimum threshold across all channel types — used to pre-filter the similarity
    # matrix before the Python loop, converting O(n²) pandas iteration to O(E) where
    # E is the number of above-threshold pairs (sparse for typical alpha >= 0.60).
    _all_alphas = [alpha]
    if alpha_switch is not None:
        _all_alphas.append(alpha_switch)
    if alpha_abundance is not None:
        _all_alphas.append(alpha_abundance)
    _min_alpha = min(_all_alphas)

    # Find above-threshold pairs in row chunks to avoid allocating a full (n×n)
    # absolute-value copy (~11 GB for n=37k features) that would trigger OOM.
    # Each chunk view is O(chunk × n) extra memory (~150 MB with chunk=512).
    _n = len(_gene_ids)
    _chunk = min(512, _n)
    _i_parts: list[np.ndarray] = []
    _j_parts: list[np.ndarray] = []
    for _start in range(0, _n, _chunk):
        _end = min(_start + _chunk, _n)
        _ci, _cj = np.where(np.abs(similarity[_start:_end]) >= _min_alpha)
        _mask = (_start + _ci) < _cj
        _i_parts.append((_start + _ci)[_mask])
        _j_parts.append(_cj[_mask])
    _i_arr = np.concatenate(_i_parts) if _i_parts else np.empty(0, dtype=np.intp)
    _j_arr = np.concatenate(_j_parts) if _j_parts else np.empty(0, dtype=np.intp)

    for i, j in zip(_i_arr, _j_arr):
        source_gene = _gene_ids[i]
        target_gene = _gene_ids[j]
        if source_gene == target_gene:
            continue
        source_type = _feat_types[i]
        target_type = _feat_types[j]
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
            "source_feature_id": _feat_ids[i],
            "target_feature_id": _feat_ids[j],
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
