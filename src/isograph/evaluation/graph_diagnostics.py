"""Stage-3 graph prior diagnostics, ablation utilities, and edge topology metrics."""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
import pandas as pd

from isograph.evaluation.metrics import module_recovery_score
from isograph.models.graph import GraphNetworkModel
from isograph.workflow.config import GraphModelConfig


def edge_topology_report(
    edge_table: pd.DataFrame,
    truth_modules: pd.DataFrame,
) -> dict:
    """Compute within-module edge precision, recall, and hub metrics.

    Works on output from any stage (Stage 1/2/3) — only requires the
    standard edge_table (source, target, weight) and truth_modules
    (gene_id, module_id).

    Returns
    -------
    dict with keys:

    Edge precision/recall (treating within-module pairs as positives):
      within_module_precision  — fraction of inferred edges that are within-module
      within_module_recall     — fraction of true within-module pairs with an inferred edge
      n_inferred_edges         — total inferred edges
      n_within_inferred        — inferred edges that are within-module
      n_possible_within        — total gene pairs sharing a module
      f1_score                 — harmonic mean of precision and recall

    Edge weight enrichment (continuous; module-label-free):
      mean_weight_within       — mean |weight| of within-module edges
      mean_weight_between      — mean |weight| of between-module edges
      weight_enrichment        — mean_weight_within / mean_weight_between (>1 = enriched)

    Hub accuracy (top-k degree genes stable across fits):
      hub_genes_top10          — list of gene_ids with highest degree in inferred network
      hub_degrees_top10        — corresponding degree values
    """
    module_map: dict[str, str] = {}
    if truth_modules is not None and not truth_modules.empty:
        module_map = dict(zip(truth_modules["gene_id"], truth_modules["module_id"]))

    # Build set of all true within-module pairs
    genes_by_module: dict = {}
    for gene, mod in module_map.items():
        genes_by_module.setdefault(mod, []).append(gene)

    true_within: set[frozenset] = set()
    for members in genes_by_module.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                true_within.add(frozenset({a, b}))

    n_possible_within = len(true_within)

    # Parse inferred edges
    inferred_edges: list[tuple[frozenset, float]] = []
    if not edge_table.empty:
        for _, row in edge_table.iterrows():
            inferred_edges.append((
                frozenset({row["source"], row["target"]}),
                float(row["weight"]),
            ))

    n_inferred = len(inferred_edges)

    # Precision / recall
    within_inferred = [(e, w) for e, w in inferred_edges if e in true_within]
    between_inferred = [(e, w) for e, w in inferred_edges if e not in true_within]

    n_within_inferred = len(within_inferred)
    precision = n_within_inferred / n_inferred if n_inferred > 0 else 0.0
    recall = n_within_inferred / n_possible_within if n_possible_within > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Weight enrichment
    within_weights = [abs(w) for _, w in within_inferred]
    between_weights = [abs(w) for _, w in between_inferred]
    mean_w_within = float(np.mean(within_weights)) if within_weights else 0.0
    mean_w_between = float(np.mean(between_weights)) if between_weights else 0.0
    # inf when all inferred edges are within-module (perfect precision); nan when no edges at all
    if mean_w_between > 0:
        weight_enrichment = mean_w_within / mean_w_between
    elif mean_w_within > 0:
        weight_enrichment = float("inf")
    else:
        weight_enrichment = float("nan")

    # Hub genes by degree
    degree_counter: dict[str, int] = {}
    for (e, _) in inferred_edges:
        a, b = tuple(e)
        degree_counter[a] = degree_counter.get(a, 0) + 1
        degree_counter[b] = degree_counter.get(b, 0) + 1
    top10 = sorted(degree_counter, key=lambda g: -degree_counter[g])[:10]

    return {
        "within_module_precision": precision,
        "within_module_recall": recall,
        "f1_score": f1,
        "n_inferred_edges": n_inferred,
        "n_within_inferred": n_within_inferred,
        "n_possible_within": n_possible_within,
        "mean_weight_within": mean_w_within,
        "mean_weight_between": mean_w_between,
        "weight_enrichment": weight_enrichment,
        "hub_genes_top10": top10,
        "hub_degrees_top10": [degree_counter.get(g, 0) for g in top10],
    }


def graph_prior_edge_report(
    gene_graph: nx.Graph,
    edge_table: pd.DataFrame,
    truth_modules: pd.DataFrame | None,
) -> dict:
    """Characterize overlap between prior graph edges and model-inferred edges.

    Parameters
    ----------
    gene_graph:
        The biological prior graph (from build_gene_graph).
    edge_table:
        Edges inferred by GraphNetworkModel (columns: source, target, weight).
    truth_modules:
        Ground-truth module assignments (columns: gene_id, module_id).
        Pass None for real data without known modules.

    Returns
    -------
    dict with keys:
      n_prior_edges, n_inferred_edges, n_overlap,
      prior_edge_inferred_fraction, inferred_edge_prior_fraction,
      within_module_prior_fraction, between_module_prior_fraction
      (last two are None when truth_modules is None).
    """
    prior_edges: set[frozenset] = {
        frozenset({u, v}) for u, v in gene_graph.edges()
    }
    inferred_edges: set[frozenset] = set()
    if not edge_table.empty:
        for _, row in edge_table.iterrows():
            inferred_edges.add(frozenset({row["source"], row["target"]}))

    n_prior = len(prior_edges)
    n_inferred = len(inferred_edges)
    n_overlap = len(prior_edges & inferred_edges)

    result: dict = {
        "n_prior_edges": n_prior,
        "n_inferred_edges": n_inferred,
        "n_overlap": n_overlap,
        "prior_edge_inferred_fraction": n_overlap / n_prior if n_prior > 0 else 0.0,
        "inferred_edge_prior_fraction": n_overlap / n_inferred if n_inferred > 0 else 0.0,
        "within_module_prior_fraction": None,
        "between_module_prior_fraction": None,
    }

    if truth_modules is not None and not truth_modules.empty and n_prior > 0:
        module_map = dict(zip(truth_modules["gene_id"], truth_modules["module_id"]))
        gene_ids_in_graph = list(gene_graph.nodes())

        # Count possible within/between pairs in the graph
        n_within_possible = 0
        n_between_possible = 0
        for i, a in enumerate(gene_ids_in_graph):
            for j in range(i + 1, len(gene_ids_in_graph)):
                b = gene_ids_in_graph[j]
                ma, mb = module_map.get(a), module_map.get(b)
                if ma is None or mb is None:
                    continue
                if ma == mb:
                    n_within_possible += 1
                else:
                    n_between_possible += 1

        within = 0
        between = 0
        within_weights: list[float] = []
        between_weights: list[float] = []
        for u, v, data in gene_graph.edges(data=True):
            mu, mv = module_map.get(u), module_map.get(v)
            if mu is None or mv is None:
                continue
            w = float(data.get("weight", 1.0))
            if mu == mv:
                within += 1
                within_weights.append(w)
            else:
                between += 1
                between_weights.append(w)

        total = within + between
        result["within_module_prior_fraction"] = within / total if total > 0 else 0.0
        result["between_module_prior_fraction"] = between / total if total > 0 else 0.0
        # Edge rate: edges per possible pair — enrichment signal for sparse graphs
        result["within_module_edge_rate"] = within / n_within_possible if n_within_possible > 0 else 0.0
        result["between_module_edge_rate"] = between / n_between_possible if n_between_possible > 0 else 0.0
        # Mean edge weight: enrichment signal for dense graphs where rate saturates at 1.0
        result["within_module_mean_weight"] = float(np.mean(within_weights)) if within_weights else 0.0
        result["between_module_mean_weight"] = float(np.mean(between_weights)) if between_weights else 0.0

    return result


def graph_ablation_report(
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    sample_table: pd.DataFrame,
    truth_modules: pd.DataFrame | None,
    ablation_configs: list[tuple[str, GraphModelConfig]],
) -> dict:
    """Run a set of ablation fits and return a comparison report.

    Parameters
    ----------
    transcript_counts:
        Raw transcript count matrix (n_transcripts, n_samples).
    transcript_table:
        Must contain columns transcript_id, gene_id.
    sample_table:
        Sample metadata.
    truth_modules:
        Ground-truth module assignments for recovery scoring.
        Pass None for real data.
    ablation_configs:
        List of (label, GraphModelConfig) pairs.

    Returns
    -------
    dict with keys:
      "ablations": list of per-ablation result dicts, each containing
        label, recovery (or None), n_edges, n_modules, calibration.
    """
    results = []
    for label, cfg in ablation_configs:
        model = GraphNetworkModel(cfg)
        artifacts = model.fit(transcript_counts, transcript_table, sample_table)
        recovery = None
        if truth_modules is not None and not truth_modules.empty:
            recovery = float(module_recovery_score(artifacts.module_table, truth_modules))
        results.append({
            "label": label,
            "recovery": recovery,
            "n_edges": len(artifacts.edge_table),
            "n_modules": artifacts.module_table["module_id"].nunique() if not artifacts.module_table.empty else 0,
            "calibration": artifacts.calibration,
        })
    return {"ablations": results}
