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


def select_alpha_switch(
    similarity: np.ndarray,
    feature_info: pd.DataFrame,
    alpha_switch_grid: list[float],
    giant_fraction_threshold: float = 0.30,
) -> tuple[float, list[dict]]:
    """Select the smallest alpha_switch that avoids a giant switch-switch component.

    Evaluates only switch-switch edges (cross-channel and abundance-abundance edges
    are suppressed) at each candidate threshold. Returns (selected_alpha, sweep_stats).

    Iterates candidates from smallest to largest; returns the first where the giant
    component fraction (largest component / total switch-connected genes) is below
    giant_fraction_threshold. Falls back to max(grid) if all candidates exceed it.
    """
    _SUPPRESS = 2.0  # alpha value that excludes all non-switch-switch edges

    sweep_stats: list[dict] = []
    selected = max(alpha_switch_grid)

    for candidate in sorted(alpha_switch_grid):
        graph, _ = project_feature_similarity_to_gene_graph(
            similarity, feature_info, _SUPPRESS,
            allow_abundance_abundance=False,
            alpha_switch=candidate,
        )
        connected = [n for n in graph.nodes if graph.degree(n) > 0]
        n_connected = len(connected)
        if n_connected == 0:
            sweep_stats.append({
                "alpha_switch": candidate,
                "n_switch_connected": 0,
                "giant_size": 0,
                "giant_fraction": 0.0,
                "n_modules_ge30": 0,
            })
            continue
        comps = sorted(nx.connected_components(graph.subgraph(connected)), key=len, reverse=True)
        giant_size = len(comps[0])
        giant_frac = giant_size / n_connected
        n_modules = sum(1 for c in comps if len(c) >= 30)
        sweep_stats.append({
            "alpha_switch": candidate,
            "n_switch_connected": n_connected,
            "giant_size": giant_size,
            "giant_fraction": round(giant_frac, 4),
            "n_modules_ge30": n_modules,
        })
        if giant_frac < giant_fraction_threshold:
            selected = candidate
            break

    return selected, sweep_stats


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


def reconstruction_to_similarity(x_recon: np.ndarray) -> np.ndarray:
    """Double-centered, L2-normalized Pearson similarity over a VAE reconstruction.

    ``x_recon`` is ``(n_samples, n_features)``; returns an ``(n_features, n_features)``
    matrix with a zero diagonal. Uses a chunked float32 matmul so peak RAM stays at
    O(n_features^2 * 4 bytes) instead of np.corrcoef's ~2x float64 overhead (which
    OOMs for n_features ~ 37k). This is the single source of truth for the edge
    similarity, shared by the VAE fit and by post-hoc multi-tier re-projection.
    """
    X = x_recon.T.astype(np.float32, copy=False)  # (n_features, n_samples)
    X = X - X.mean(axis=1, keepdims=True)
    X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=1)
    X /= np.where(norms > 1e-12, norms, 1.0)[:, None]
    n = X.shape[0]
    sim = np.empty((n, n), dtype=np.float32)
    _chunk = min(512, n)
    for _start in range(0, n, _chunk):
        _end = min(_start + _chunk, n)
        sim[_start:_end] = X[_start:_end] @ X.T
    np.fill_diagonal(sim, 0.0)
    return sim


def project_feature_similarity_to_gene_graph(
    similarity: np.ndarray,
    feature_info: pd.DataFrame,
    alpha: float,
    allow_abundance_abundance: bool = False,
    alpha_switch: float | None = None,
    alpha_abundance: float | None = None,
    gene_reliability: dict[str, float] | None = None,
    node_stats: dict[str, dict[str, float]] | None = None,
    switch_only: bool = False,
) -> tuple[nx.Graph, list[dict[str, object]]]:
    """Aggregate feature-channel similarities into a gene-level graph.

    Switch↔abundance cross-channel edges between dual-channel genes are always
    suppressed (use the dedicated switch-switch edge instead). Abundance↔abundance
    edges between dual-channel genes are included only when allow_abundance_abundance
    is True; use alpha_abundance (or alpha_abundance_grid via select_alpha_abundance)
    to calibrate the per-channel threshold and avoid giant components.

    ``gene_reliability`` (gene_id -> weight in [0, 1]) downweights switch-switch
    edges by the geometric mean of the two genes' switch reliabilities
    (``edge_weight = association * sqrt(r_a * r_b)``). Degradation-dominated genes
    thus drop their switch edges below threshold and fall back to the abundance
    channel. Abundance-involving edges are not reweighted (abundance is
    degradation-robust). Applied before thresholding so unreliable edges are
    filtered out.

    ``node_stats`` (optional, populated in place) collects per-gene connectivity
    evidence so a downstream node-fate diagnostic can explain why a target gene
    is absent from every module. For each gene that participates in at least one
    candidate pair, it records ``max_abs_assoc`` (the strongest |raw similarity|
    to any partner, regardless of channel) and ``max_switch_assoc`` (the strongest
    |raw switch-switch similarity|, *before* reliability downweighting). Both are
    capped at the pre-filter floor ``min(alpha, alpha_switch, alpha_abundance)``:
    genes whose best association falls below that floor never enter the loop and
    are simply omitted from ``node_stats`` (caller treats them as "< min_alpha").
    """
    gene_ids = sorted(feature_info["gene_id"].astype(str).unique())
    genes_with_switch = set(
        feature_info.loc[feature_info["feature_type"].astype(str) == "switch", "gene_id"].astype(str)
    )
    graph = nx.Graph()
    graph.add_nodes_from(gene_ids)
    best: dict[tuple[str, str], dict[str, object]] = {}

    # Per-gene strongest raw association seen across candidate pairs (>= min_alpha),
    # for the optional node-fate diagnostic. Tracked only when node_stats is given.
    _max_assoc: dict[str, float] = {}
    _max_switch_assoc: dict[str, float] = {}

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
        # Pure isoform-switch ablation: keep only switch-switch edges (drop every
        # abundance-involving edge, cross-channel included). The VAE still denoises
        # the full multiplex feature matrix; only the gene graph is switch-restricted.
        if switch_only and not (source_type == "switch" and target_type == "switch"):
            continue
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
        # Record per-gene raw-association maxima (pre-threshold, pre-reliability)
        # for the node-fate diagnostic. Switch-switch is tracked separately so the
        # caller can tell "no partner at all" from "partner lost to downweighting".
        if node_stats is not None:
            raw_abs = abs(weight)
            if raw_abs > _max_assoc.get(source_gene, 0.0):
                _max_assoc[source_gene] = raw_abs
            if raw_abs > _max_assoc.get(target_gene, 0.0):
                _max_assoc[target_gene] = raw_abs
            if source_type == "switch" and target_type == "switch":
                if raw_abs > _max_switch_assoc.get(source_gene, 0.0):
                    _max_switch_assoc[source_gene] = raw_abs
                if raw_abs > _max_switch_assoc.get(target_gene, 0.0):
                    _max_switch_assoc[target_gene] = raw_abs
        # Resolve per-channel threshold.
        if alpha_switch is not None and source_type == "switch" and target_type == "switch":
            effective_alpha = alpha_switch
        elif alpha_abundance is not None and source_type == "abundance" and target_type == "abundance":
            effective_alpha = alpha_abundance
        else:
            effective_alpha = alpha
        # Degradation-aware reliability downweighting of switch-switch edges only.
        if gene_reliability is not None and source_type == "switch" and target_type == "switch":
            r_s = gene_reliability.get(source_gene, 1.0)
            r_t = gene_reliability.get(target_gene, 1.0)
            weight *= float(np.sqrt(max(r_s, 0.0) * max(r_t, 0.0)))
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

    if node_stats is not None:
        for gene_id in _max_assoc.keys() | _max_switch_assoc.keys():
            node_stats[gene_id] = {
                "max_abs_assoc": _max_assoc.get(gene_id, float("nan")),
                "max_switch_assoc": _max_switch_assoc.get(gene_id, float("nan")),
            }

    return graph, edge_rows
