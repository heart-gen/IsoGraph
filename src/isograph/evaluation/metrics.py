"""Benchmark metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from isograph.models.base import FitArtifacts


def calibration_metrics(artifacts_by_fixture: dict[str, FitArtifacts]) -> dict:
    """Aggregate calibration dicts from fit artifacts keyed by fixture name."""
    rows = []
    for fixture_name, arts in artifacts_by_fixture.items():
        if arts.calibration is not None:
            rows.append({"fixture": fixture_name, **arts.calibration})
    return {"calibration_by_fixture": rows}


def module_recovery_score(predicted: pd.DataFrame, truth: pd.DataFrame) -> float:
    if predicted.empty or truth.empty:
        return 0.0
    predicted_groups = [set(group["gene_id"]) for _, group in predicted.groupby("module_id")]
    truth_groups = [set(group["gene_id"]) for _, group in truth.groupby("module_id")]
    total = 0.0
    for truth_group in truth_groups:
        best = 0.0
        for predicted_group in predicted_groups:
            union = len(truth_group | predicted_group)
            if union == 0:
                continue
            best = max(best, len(truth_group & predicted_group) / union)
        total += best
    return total / len(truth_groups)


def role_aware_recall(
    predicted_modules: pd.DataFrame,
    truth_channel_role: pd.DataFrame,
) -> dict[str, float]:
    """Per-role recall for switch_only, abundance_only, coupled, discordant.

    For each channel role, computes the fraction of genes with that role (across
    all truth modules) that appear in their best-matching predicted module.
    Best-match is determined by Jaccard IoU, same as module_recovery_score.
    Background genes (truth_role == 'background') are excluded.
    """
    roles = ("switch_only", "abundance_only", "coupled", "discordant")
    if predicted_modules.empty or truth_channel_role.empty:
        return {r: 0.0 for r in roles}

    truth_role_active = truth_channel_role[truth_channel_role["truth_role"] != "background"]
    if truth_role_active.empty:
        return {r: 0.0 for r in roles}

    truth_module_sets: dict[object, set[str]] = {
        mid: set(grp["gene_id"].astype(str)) for mid, grp in truth_role_active.groupby("module_id")
    }
    pred_module_sets: list[set[str]] = [
        set(grp["gene_id"].astype(str)) for _, grp in predicted_modules.groupby("module_id")
    ]
    gene_role: dict[str, str] = dict(
        zip(
            truth_role_active["gene_id"].astype(str),
            truth_role_active["truth_role"].astype(str),
            strict=False,
        )
    )

    role_recovered: dict[str, int] = {r: 0 for r in roles}
    role_total: dict[str, int] = {r: 0 for r in roles}

    for truth_genes in truth_module_sets.values():
        # Best-matching predicted module by IoU
        best_pred: set[str] = set()
        best_iou = 0.0
        for pred_genes in pred_module_sets:
            union = len(truth_genes | pred_genes)
            if union == 0:
                continue
            iou = len(truth_genes & pred_genes) / union
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_genes

        for gene in truth_genes:
            role = gene_role.get(gene)
            if role not in role_total:
                continue
            role_total[role] += 1
            if gene in best_pred:
                role_recovered[role] += 1

    return {r: (role_recovered[r] / role_total[r] if role_total[r] > 0 else 0.0) for r in roles}


def giant_component_fraction(edge_table: pd.DataFrame, n_genes: int) -> float:
    """Fraction of genes in the largest connected component of the edge graph."""
    if edge_table.empty or n_genes == 0:
        return 0.0
    g = nx.Graph()
    for _, row in edge_table[["source", "target"]].iterrows():
        g.add_edge(str(row["source"]), str(row["target"]))
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    return len(components[0]) / n_genes if components else 0.0
