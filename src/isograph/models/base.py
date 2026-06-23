"""Model interfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from isograph.features.channels import feature_sample_columns

_log = logging.getLogger(__name__)


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


def _kcore_module(graph: nx.Graph, nodes: set, k: int) -> set:
    """Background/grey rejection: the k-core of a module's induced subgraph.

    Iteratively removes genes whose degree *within the module* is < k, until all
    remaining genes have intra-module degree >= k. Weakly-attached background
    genes (joined to a module by only one or two spurious edges) are dropped to
    grey rather than diluting the module; densely co-connected true module genes
    survive. Mirrors WGCNA leaving low-connectivity genes unassigned.
    """
    sub = graph.subgraph(nodes).copy()
    sub.remove_edges_from(nx.selfloop_edges(sub))
    while sub.number_of_nodes():
        low = [n for n, deg in sub.degree() if deg < k]
        if not low:
            break
        sub.remove_nodes_from(low)
    return set(sub.nodes())


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

        grey_k = getattr(self.config, "grey_min_intra_degree", 0)

        rows = []
        for module_index, nodes in enumerate(communities):
            if grey_k > 0:
                nodes = _kcore_module(pos_graph, nodes, grey_k)
            if len(nodes) < self.config.min_module_size:
                continue
            for gene_id in sorted(nodes):
                rows.append({"gene_id": gene_id, "module_id": f"M{module_index:03d}"})
        return pd.DataFrame(rows)

    def _detect_communities(self, pos_graph: nx.Graph) -> list[set]:
        """Return node communities, largest first.

        Module-detection backend, selected by config:

        * ``leiden_max_giant_frac`` set (collapse fix C) -> *data-driven*
          resolution: sweep increasing Leiden resolutions
          (``leiden_resolution_grid`` or a default geometric grid) and pick the
          smallest whose largest community is <= that fraction of all genes. This
          removes the hand-picked-resolution knob -- the chief module-collapse
          defect, since ``connected_components`` fuses everything into one giant
          module via single bridge genes on a dense reconstruction-similarity graph.
        * ``leiden_resolution`` set (without a cap) -> seeded, edge-weighted Leiden
          at that fixed resolution.
        * neither set -> legacy ``connected_components`` (preserves old run hashes;
          default).

        Leiden is edge-weighted and seeded on ``config.random_state`` for
        determinism. The chosen backend/resolution and achieved giant fraction are
        recorded on ``self._leiden_selection`` so the partition is auditable and
        reproducible from ``(config, seed)``.
        """
        n_nodes = pos_graph.number_of_nodes()
        resolution = getattr(self.config, "leiden_resolution", None)
        max_giant_frac = getattr(self.config, "leiden_max_giant_frac", None)

        if resolution is not None or max_giant_frac is not None:
            try:
                import igraph as ig
                import leidenalg
            except ImportError:
                _log.warning(
                    "Leiden module detection requested (leiden_resolution/"
                    "leiden_max_giant_frac) but igraph/leidenalg are not installed; "
                    "falling back to connected components."
                )
            else:
                nodes_list = list(pos_graph.nodes())
                node_to_idx = {n: i for i, n in enumerate(nodes_list)}
                edges = [(node_to_idx[u], node_to_idx[v]) for u, v in pos_graph.edges()]
                weights = [float(d.get("weight", 1.0)) for _, _, d in pos_graph.edges(data=True)]
                seed = int(getattr(self.config, "random_state", 0) or 0)
                ig_graph = ig.Graph(n=len(nodes_list), edges=edges)

                def _partition(res: float) -> list[set]:
                    part = leidenalg.find_partition(
                        ig_graph,
                        leidenalg.RBConfigurationVertexPartition,
                        weights=weights or None,
                        resolution_parameter=res,
                        seed=seed,
                    )
                    comms = [{nodes_list[v] for v in c} for c in part]
                    return sorted(comms, key=len, reverse=True)

                def _giant_frac(comms: list[set]) -> float:
                    return (len(comms[0]) / n_nodes) if comms and n_nodes else 0.0

                if max_giant_frac is not None:
                    grid = getattr(self.config, "leiden_resolution_grid", None)
                    if not grid:
                        base_r = resolution if resolution is not None else 1.0
                        grid = [base_r * (2 ** i) for i in range(6)]
                    grid = sorted({float(r) for r in grid})
                    tried: list[dict] = []
                    chosen: tuple[float, float, list[set]] | None = None
                    best: tuple[float, float, list[set]] | None = None
                    for r in grid:
                        comms = _partition(r)
                        gf = _giant_frac(comms)
                        tried.append({"resolution": r, "giant_fraction": gf, "n_modules": len(comms)})
                        if best is None or gf < best[1]:
                            best = (r, gf, comms)
                        if gf <= max_giant_frac:
                            chosen = (r, gf, comms)
                            break
                    cap_met = chosen is not None
                    r_sel, gf_sel, comms_sel = chosen if chosen is not None else best  # type: ignore[misc]
                    if not cap_met:
                        _log.warning(
                            "leiden_max_giant_frac=%.3f not reached on the resolution grid "
                            "(best giant fraction %.3f at resolution %.4g); using best effort.",
                            max_giant_frac, gf_sel, r_sel,
                        )
                    self._leiden_selection = {
                        "mode": "auto",
                        "resolution": r_sel,
                        "giant_fraction": gf_sel,
                        "n_modules": len(comms_sel),
                        "cap": max_giant_frac,
                        "cap_met": cap_met,
                        "grid_tried": tried,
                        "seed": seed,
                    }
                    return comms_sel

                comms = _partition(float(resolution))
                self._leiden_selection = {
                    "mode": "fixed",
                    "resolution": float(resolution),
                    "giant_fraction": _giant_frac(comms),
                    "n_modules": len(comms),
                    "cap": None,
                    "cap_met": None,
                    "seed": seed,
                }
                return comms

        comms = sorted(nx.connected_components(pos_graph), key=len, reverse=True)
        self._leiden_selection = {
            "mode": "connected_components",
            "resolution": None,
            "giant_fraction": (len(comms[0]) / n_nodes) if comms and n_nodes else 0.0,
            "n_modules": len(comms),
            "cap": None,
            "cap_met": None,
        }
        return comms
