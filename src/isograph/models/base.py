"""Model interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from isograph.features.channels import feature_sample_columns


@dataclass
class FitArtifacts:
    module_table: pd.DataFrame
    edge_table: pd.DataFrame
    trait_table: pd.DataFrame
    feature_scores: pd.DataFrame
    calibration: dict | None = None
    checkpoint_path: Path | None = None
    eigengene_table: pd.DataFrame | None = None
    module_gene_roles: pd.DataFrame | None = None


def _module_feature_subset(feature_scores: pd.DataFrame, genes: pd.Series | list[str]) -> pd.DataFrame:
    return feature_scores.loc[feature_scores["gene_id"].isin(set(genes))]


def _feature_matrix(frame: pd.DataFrame, sample_ids: list[str]) -> np.ndarray:
    return frame[sample_ids].to_numpy(dtype=float)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return 0.0
    left = left[mask]
    right = right[mask]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _first_component_reference(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if not np.isfinite(matrix).all():
        row_means = np.nanmean(np.where(np.isfinite(matrix), matrix, np.nan), axis=1)
        row_means = np.where(np.isfinite(row_means), row_means, 0.0)
        matrix = np.where(np.isfinite(matrix), matrix, row_means[:, None])
    if matrix.shape[0] == 1:
        return matrix[0]
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    if np.all(np.std(centered, axis=1) <= 1e-12):
        return centered.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return centered.mean(axis=0)
    return vh[0]


def compute_trait_associations(
    module_table: pd.DataFrame,
    feature_scores: pd.DataFrame,
    sample_table: pd.DataFrame,
    trait_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute eigengene-trait associations for all modules.

    Numeric columns → Pearson correlation (effect = r).
    Binary categorical columns (exactly 2 unique values) → Welch t-test
    (effect = group1_mean - group0_mean, groups ordered by sorted category label).
    Columns absent from sample_table or with >2 categories are skipped.
    """
    sample_ids = (
        sample_table["sample_id"].tolist()
        if "sample_id" in sample_table.columns
        else list(range(len(sample_table)))
    )
    if module_table.empty:
        return (
            pd.DataFrame(columns=["module_id", "trait", "effect", "pvalue"]),
            pd.DataFrame(columns=["module_id"] + sample_ids),
        )
    rows = []
    eigengene_rows: dict[str, list] = {}
    for module_id in sorted(module_table["module_id"].unique()):
        genes = module_table.loc[module_table["module_id"] == module_id, "gene_id"]
        subset = _module_feature_subset(feature_scores, genes)
        eigengene = _feature_matrix(subset, sample_ids).mean(axis=0)
        eigengene_rows[module_id] = eigengene.tolist()
        for col in trait_columns:
            if col not in sample_table.columns:
                continue
            series = sample_table[col]
            if pd.api.types.is_numeric_dtype(series):
                vals = series.to_numpy(dtype=float)
                mask = np.isfinite(vals) & np.isfinite(eigengene)
                if mask.sum() < 10:
                    continue
                effect, pvalue = stats.pearsonr(eigengene[mask], vals[mask])
                rows.append({"module_id": module_id, "trait": col, "effect": float(effect), "pvalue": float(pvalue)})
            else:
                cats = sorted(series.dropna().unique())
                if len(cats) != 2:
                    continue
                cat0, cat1 = cats
                grp0 = eigengene[(series == cat0).to_numpy()]
                grp1 = eigengene[(series == cat1).to_numpy()]
                if len(grp0) < 2 or len(grp1) < 2:
                    continue
                ttest = stats.ttest_ind(grp0, grp1, equal_var=False)
                effect = float(grp1.mean() - grp0.mean())
                rows.append({"module_id": module_id, "trait": col, "effect": effect, "pvalue": float(ttest.pvalue)})
    eigengene_table = (
        pd.DataFrame(eigengene_rows, index=sample_ids)
        .T.reset_index()
        .rename(columns={"index": "module_id"})
    )
    return pd.DataFrame(rows), eigengene_table


def compute_module_gene_roles(
    module_table: pd.DataFrame,
    feature_scores: pd.DataFrame,
    sample_table: pd.DataFrame,
    min_abs_r: float = 0.2,
) -> pd.DataFrame:
    """Classify each module gene by switch/abundance channel participation."""
    sample_ids = feature_sample_columns(feature_scores)
    if module_table.empty or not sample_ids:
        return pd.DataFrame(
            columns=[
                "module_id",
                "gene_id",
                "module_role",
                "switch_r",
                "abundance_r",
                "switch_abundance_r",
                "switch_active",
                "abundance_active",
            ]
        )

    rows: list[dict[str, object]] = []
    for module_id in sorted(module_table["module_id"].unique()):
        genes = module_table.loc[module_table["module_id"] == module_id, "gene_id"]
        subset = _module_feature_subset(feature_scores, genes)
        if subset.empty:
            continue
        channel_references: dict[str, np.ndarray] = {}
        for feature_type, frame in subset.groupby("feature_type"):
            channel_references[str(feature_type)] = _first_component_reference(
                _feature_matrix(frame, sample_ids)
            )
        for gene_id in sorted(set(genes)):
            gene_features = subset.loc[subset["gene_id"] == gene_id]
            channel_r: dict[str, float] = {}
            channel_values: dict[str, np.ndarray] = {}
            for feature_type, frame in gene_features.groupby("feature_type"):
                values = _feature_matrix(frame, sample_ids).mean(axis=0)
                key = str(feature_type)
                channel_values[key] = values
                reference = channel_references.get(key)
                channel_r[key] = 0.0 if reference is None else _safe_corr(values, reference)
            switch_r = channel_r.get("switch", np.nan)
            abundance_r = channel_r.get("abundance", np.nan)
            switch_abundance_r = np.nan
            if "switch" in channel_values and "abundance" in channel_values:
                switch_abundance_r = _safe_corr(channel_values["switch"], channel_values["abundance"])
            switch_active = bool(np.isfinite(switch_r) and abs(switch_r) >= min_abs_r)
            abundance_active = bool(np.isfinite(abundance_r) and abs(abundance_r) >= min_abs_r)
            if switch_active and abundance_active:
                role = (
                    "coupled"
                    if np.isfinite(switch_abundance_r) and switch_abundance_r >= min_abs_r
                    else "discordant"
                )
            elif switch_active:
                role = "switch_only"
            elif abundance_active:
                role = "abundance_only"
            else:
                role = "inactive"
            rows.append(
                {
                    "module_id": module_id,
                    "gene_id": gene_id,
                    "module_role": role,
                    "switch_r": switch_r,
                    "abundance_r": abundance_r,
                    "switch_abundance_r": switch_abundance_r,
                    "switch_active": switch_active,
                    "abundance_active": abundance_active,
                }
            )
    return pd.DataFrame(rows)


class NetworkModel:
    def fit(self, *args, **kwargs) -> FitArtifacts:
        raise NotImplementedError

    def _module_table(self, graph: nx.Graph) -> pd.DataFrame:
        """Assign genes to modules from the (positive-weight) network.

        Edges with non-positive weight are dropped before module detection.
        When ``config.leiden_resolution`` is set and ``igraph``/``leidenalg`` are
        available, Leiden community detection is used; otherwise it falls back to
        connected components.
        """
        pos_graph = nx.Graph()
        pos_graph.add_nodes_from(graph.nodes())
        pos_graph.add_edges_from(
            (u, v, d) for u, v, d in graph.edges(data=True) if d.get("weight", 1.0) > 0
        )

        communities = self._detect_communities(pos_graph)

        rows = []
        for module_index, nodes in enumerate(communities):
            if len(nodes) < self.config.min_module_size:
                continue
            for gene_id in sorted(nodes):
                rows.append({"gene_id": gene_id, "module_id": f"M{module_index:03d}"})
        return pd.DataFrame(rows)

    def _detect_communities(self, pos_graph: nx.Graph) -> list[set]:
        """Return node communities, largest first.

        Uses Leiden when ``config.leiden_resolution`` is set and the optional
        ``igraph``/``leidenalg`` dependencies are installed, otherwise falls back
        to connected components.
        """
        resolution = getattr(self.config, "leiden_resolution", None)
        if resolution is not None:
            try:
                import igraph as ig
                import leidenalg

                nodes_list = list(pos_graph.nodes())
                node_to_idx = {n: i for i, n in enumerate(nodes_list)}
                edges = [(node_to_idx[u], node_to_idx[v]) for u, v in pos_graph.edges()]
                ig_graph = ig.Graph(n=len(nodes_list), edges=edges)
                partition = leidenalg.find_partition(
                    ig_graph,
                    leidenalg.RBConfigurationVertexPartition,
                    resolution_parameter=resolution,
                )
                communities = [{nodes_list[v] for v in community} for community in partition]
                return sorted(communities, key=len, reverse=True)
            except ImportError:
                pass

        return sorted(nx.connected_components(pos_graph), key=len, reverse=True)
