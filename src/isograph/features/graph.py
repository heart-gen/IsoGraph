"""Gene-level graph construction and Laplacian smoothing for Stage-3 graph priors."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
from scipy import linalg


def build_gene_graph(
    switch_matrix: np.ndarray,
    gene_ids: list[str],
    transcript_table: pd.DataFrame,
    edge_types: list[str],
    corr_threshold: float = 0.3,
) -> nx.Graph:
    """Build a weighted gene-level graph from biological relationships.

    Parameters
    ----------
    switch_matrix:
        Residualized gene switch coordinates, shape (n_genes, n_samples).
    gene_ids:
        Ordered gene identifiers matching axis-0 of switch_matrix.
    transcript_table:
        Must contain columns transcript_id and gene_id.
    edge_types:
        Subset of {"corr", "same_gene"}.
    corr_threshold:
        Minimum absolute Pearson r for a corr-type edge.

    Returns
    -------
    nx.Graph with gene_id node labels.  Edge attribute ``weight`` is absolute r for
    corr edges and 1.0 for same_gene edges.
    """
    graph = nx.Graph()
    graph.add_nodes_from(gene_ids)

    if "corr" in edge_types and len(gene_ids) >= 2:
        _add_corr_edges(graph, switch_matrix, gene_ids, corr_threshold)

    if "same_gene" in edge_types:
        _add_same_gene_edges(graph, transcript_table, gene_ids)

    return graph


def _add_corr_edges(
    graph: nx.Graph,
    switch_matrix: np.ndarray,
    gene_ids: list[str],
    threshold: float,
) -> None:
    n = len(gene_ids)
    if n < 2 or switch_matrix.shape[1] < 2:
        return
    R = np.corrcoef(switch_matrix)  # (n_genes, n_genes)
    np.fill_diagonal(R, 0.0)
    rows, cols = np.where(np.abs(R) >= threshold)
    for i, j in zip(rows, cols):
        if i < j:
            w = float(abs(R[i, j]))
            if graph.has_edge(gene_ids[i], gene_ids[j]):
                # Keep higher weight when multiple edge types add the same pair
                graph[gene_ids[i]][gene_ids[j]]["weight"] = max(
                    graph[gene_ids[i]][gene_ids[j]]["weight"], w
                )
            else:
                graph.add_edge(gene_ids[i], gene_ids[j], weight=w)


def _add_same_gene_edges(
    graph: nx.Graph,
    transcript_table: pd.DataFrame,
    gene_ids: list[str],
) -> None:
    gene_id_set = set(gene_ids)
    # Find transcripts that map to multiple genes (read-through / multi-gene rows)
    # In current fixtures each transcript maps to exactly one gene, so this is a no-op.
    multi = (
        transcript_table.groupby("transcript_id")["gene_id"]
        .apply(list)
        .reset_index()
    )
    for _, row in multi.iterrows():
        genes = [g for g in row["gene_id"] if g in gene_id_set]
        for a_idx in range(len(genes)):
            for b_idx in range(a_idx + 1, len(genes)):
                a, b = genes[a_idx], genes[b_idx]
                if not graph.has_edge(a, b):
                    graph.add_edge(a, b, weight=1.0)


def graph_laplacian(
    graph: nx.Graph,
    gene_ids: list[str],
    normalized: bool = True,
) -> np.ndarray:
    """Compute the graph Laplacian as a dense numpy array.

    Parameters
    ----------
    graph:
        Weighted gene graph (edge attribute ``weight`` used).
    gene_ids:
        Ordered list of gene identifiers; defines the row/column order.
    normalized:
        When True, return the symmetric normalized Laplacian
        ``L = I - D^{-1/2} A D^{-1/2}`` (eigenvalues in [0, 2]).
        When False, return the combinatorial Laplacian ``L = D - A``.

    Isolated nodes (degree 0) are handled safely: their row and column
    in the normalized Laplacian are zero (no smoothing for isolated genes).
    """
    n = len(gene_ids)
    index = {g: i for i, g in enumerate(gene_ids)}
    A = np.zeros((n, n))
    for u, v, data in graph.edges(data=True):
        if u in index and v in index:
            w = float(data.get("weight", 1.0))
            A[index[u], index[v]] = w
            A[index[v], index[u]] = w

    degrees = A.sum(axis=1)

    if not normalized:
        return np.diag(degrees) - A

    # Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    safe_deg = np.where(degrees > 0, degrees, 1.0)  # avoid divide-by-zero; result zeroed out below
    inv_sqrt_d = np.where(degrees > 0, 1.0 / np.sqrt(safe_deg), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt_d)
    return np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt


def laplacian_smooth(
    matrix: np.ndarray,
    L: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply implicit Euler graph-Laplacian smoothing.

    Computes ``X_smooth = (I + gamma * L)^{-1} X`` via a single linear-system
    solve.  When ``gamma == 0`` the input is returned unchanged (fast path,
    no allocation).

    Parameters
    ----------
    matrix:
        Gene expression / switch matrix, shape (n_genes, n_samples).
    L:
        Graph Laplacian, shape (n_genes, n_genes).
    gamma:
        Smoothing strength ≥ 0.  Larger values increase diffusion across
        graph neighbors.
    """
    if gamma == 0.0:
        return matrix
    n = matrix.shape[0]
    M = np.eye(n) + gamma * L  # symmetric PSD; eigenvalues ≥ 1
    # Solve M @ X_smooth.T = matrix.T  →  X_smooth = (M^{-1} matrix.T).T
    return linalg.solve(M, matrix, assume_a="pos")
