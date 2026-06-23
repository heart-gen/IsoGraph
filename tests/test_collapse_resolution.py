"""Collapse fix C: data-driven Leiden resolution selection in _detect_communities.

A dense reconstruction-similarity graph is a single connected component, so the
legacy connected_components backend collapses it into one giant module. The
giant-fraction-capped sweep must select a resolution that breaks the giant apart
while staying deterministic and node-conserving.
"""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from isograph.models.base import NetworkModel

pytest.importorskip("igraph")
pytest.importorskip("leidenalg")


def _ring_of_cliques(n_cliques: int = 4, clique_size: int = 10) -> nx.Graph:
    """One connected component: cliques joined in a ring by single bridge edges.

    connected_components sees a single 40-node giant; the modular structure only
    surfaces under resolution-tuned Leiden.
    """
    g = nx.Graph()
    cliques = []
    for c in range(n_cliques):
        nodes = [f"g{c}_{i}" for i in range(clique_size)]
        cliques.append(nodes)
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                g.add_edge(u, v, weight=1.0)
    for c in range(n_cliques):
        nxt = (c + 1) % n_cliques
        g.add_edge(cliques[c][0], cliques[nxt][0], weight=0.05)  # weak bridge
    return g


def _model(**cfg) -> NetworkModel:
    defaults = dict(
        leiden_resolution=None,
        leiden_max_giant_frac=None,
        leiden_resolution_grid=None,
        random_state=13,
    )
    defaults.update(cfg)
    m = NetworkModel()
    m.config = SimpleNamespace(**defaults)
    return m


def _giant_frac(comms, n):
    return len(max(comms, key=len)) / n


def test_default_backend_is_connected_components_giant():
    g = _ring_of_cliques()
    m = _model()
    comms = m._detect_communities(g)
    assert len(comms) == 1  # one giant component
    assert m._leiden_selection["mode"] == "connected_components"
    assert m._leiden_selection["giant_fraction"] == pytest.approx(1.0)


def test_giant_frac_cap_breaks_the_giant():
    g = _ring_of_cliques()
    n = g.number_of_nodes()
    m = _model(leiden_max_giant_frac=0.3)
    comms = m._detect_communities(g)
    sel = m._leiden_selection
    assert sel["mode"] == "auto"
    assert sel["cap_met"] is True
    assert sel["giant_fraction"] <= 0.3
    assert _giant_frac(comms, n) <= 0.3
    assert len(comms) >= 4
    # node-conserving: every gene assigned exactly once
    union = set().union(*comms)
    assert union == set(g.nodes())
    assert sum(len(c) for c in comms) == n
    assert sel["resolution"] is not None


def test_auto_selection_is_deterministic():
    g = _ring_of_cliques()
    a = _model(leiden_max_giant_frac=0.3)
    b = _model(leiden_max_giant_frac=0.3)
    ca = sorted(tuple(sorted(c)) for c in a._detect_communities(g))
    cb = sorted(tuple(sorted(c)) for c in b._detect_communities(g))
    assert ca == cb
    assert a._leiden_selection["resolution"] == b._leiden_selection["resolution"]


def test_fixed_resolution_records_mode_fixed():
    g = _ring_of_cliques()
    m = _model(leiden_resolution=1.0)
    m._detect_communities(g)
    sel = m._leiden_selection
    assert sel["mode"] == "fixed"
    assert sel["resolution"] == pytest.approx(1.0)
    assert sel["cap"] is None


def test_unreachable_cap_falls_back_best_effort():
    g = _ring_of_cliques()
    # impossible cap on a 40-node graph (smallest module is a 10-node clique = 0.25)
    m = _model(leiden_max_giant_frac=0.01)
    m._detect_communities(g)
    sel = m._leiden_selection
    assert sel["mode"] == "auto"
    assert sel["cap_met"] is False
    assert sel["giant_fraction"] > 0.01  # best effort, cap not reached
