"""Stage-3 graph prior diagnostics and ablation utilities."""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
import pandas as pd

from isograph.evaluation.metrics import module_recovery_score
from isograph.models.graph import GraphNetworkModel
from isograph.workflow.config import GraphModelConfig


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
