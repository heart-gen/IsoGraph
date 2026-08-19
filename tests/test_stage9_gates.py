"""Stage 9B promotion gate tests — multiplex edge policy calibration.

Covers Stage 9B checklist items:
  - Multiplex fixtures (toy/medium/noisy/large) load correctly with all truth tables
  - allow_abundance_abundance=True + alpha_abundance_grid auto-selection does not
    produce a giant component (fraction < 0.50) on medium/noisy/large fixtures
  - Module recovery on multiplex_v1 suite meets calibration-locked gates
  - Abundance recall (role_aware_recall["abundance_only"]) meets calibration-locked gates
  - Stage 6 gate tests continue passing (run test_stage6_gates.py separately)

NOTE: toy_multiplex_v1 (40 genes) is excluded from the giant-component check.
At calibrated thresholds the toy fixture is too small to avoid a giant component
across all backends — this is expected and not a regression.

CALIBRATION PROCESS
-------------------
1. Run: isograph benchmark --config-name stage9_multiplex_vae
2. Run: isograph benchmark --config-name stage9_multiplex_graph
3. Run: isograph benchmark --config-name stage9_multiplex_latent
4. Read artifacts/reports/stage9_multiplex_{vae,graph,latent}-benchmark.json
5. Set each _GATE_* = observed_recovery − 0.10
   Set each _GATE_AB_RECALL_* = observed_abundance_only_recall − 0.10
6. Commit the updated constants and re-run this file.
7. Also run pytest tests/test_stage6_gates.py -v to verify no regression.

ALPHA GRIDS AND SELECTION
-------------------------
select_alpha_abundance uses a no-merge criterion: builds a baseline gene graph
without abundance-abundance edges and returns the smallest alpha_abundance where
no two baseline switch-based modules are merged (directly or transitively) by the
new edges. Falls back to max(grid) when all candidates cause merging.

VAE: alpha_abundance_grid = [0.60..0.90] — latent-space abundance correlations
are strong; the no-merge criterion typically selects 0.70–0.85.
Graph/Latent: alpha and alpha_switch lowered to 0.02 (partial correlations in
high-dimensional multiplex feature space are smaller than in switch-only mode);
grid = [0.05..0.30], typically selects 0.05.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_skip_no_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")

# ---------------------------------------------------------------------------
# Calibration-locked recovery gates.
# All None until first calibration benchmark run completes.
# After calibration: set to observed_recovery − 0.10 (Stage 6 methodology).
# ---------------------------------------------------------------------------
_GATE_TOY_VAE: float | None = 0.400
_GATE_MEDIUM_VAE: float | None = 0.895
_GATE_NOISY_VAE: float | None = 0.883
_GATE_LARGE_VAE: float | None = 0.896

_GATE_MEDIUM_GRAPH: float | None = 0.702
_GATE_NOISY_GRAPH: float | None = 0.715

# Abundance-only recall gates — new in Stage 9B.
# Set to observed_abundance_only_recall − 0.10 after calibration.
_GATE_AB_RECALL_MEDIUM_VAE: float | None = 0.900
_GATE_AB_RECALL_NOISY_VAE: float | None = 0.900
_GATE_AB_RECALL_LARGE_VAE: float | None = 0.900

# ---------------------------------------------------------------------------
# Per-fixture configs (VAE)
# ---------------------------------------------------------------------------
from isograph.workflow.config import GraphModelConfig, VaeModelConfig

_ALPHA_ABUNDANCE_GRID_VAE = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
_ALPHA_ABUNDANCE_GRID_GRAPH_LATENT = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

_TOY_VAE_CONFIG = VaeModelConfig(
    latent_dim=4,
    hidden_dim=64,
    n_hidden_layers=2,
    n_epochs=500,
    beta=0.5,
    alpha=0.70,
    alpha_switch=0.70,
    allow_abundance_abundance=True,
    alpha_abundance_grid=_ALPHA_ABUNDANCE_GRID_VAE,
    min_module_size=2,
    random_state=7,
)
_MEDIUM_VAE_CONFIG = VaeModelConfig(
    latent_dim=12,
    hidden_dim=128,
    n_hidden_layers=2,
    n_epochs=500,
    beta=0.5,
    alpha=0.70,
    alpha_switch=0.70,
    allow_abundance_abundance=True,
    alpha_abundance_grid=_ALPHA_ABUNDANCE_GRID_VAE,
    min_module_size=2,
    random_state=7,
)
_NOISY_VAE_CONFIG = VaeModelConfig(
    latent_dim=8,
    hidden_dim=128,
    n_hidden_layers=2,
    n_epochs=500,
    beta=0.5,
    alpha=0.70,
    alpha_switch=0.70,
    allow_abundance_abundance=True,
    alpha_abundance_grid=_ALPHA_ABUNDANCE_GRID_VAE,
    min_module_size=2,
    random_state=7,
)
_LARGE_VAE_CONFIG = VaeModelConfig(
    latent_dim=10,
    hidden_dim=128,
    n_hidden_layers=2,
    n_epochs=500,
    beta=0.5,
    alpha=0.70,
    alpha_switch=0.70,
    allow_abundance_abundance=True,
    alpha_abundance_grid=_ALPHA_ABUNDANCE_GRID_VAE,
    min_module_size=2,
    random_state=7,
)

# Per-fixture configs (graph)
_MEDIUM_GRAPH_CONFIG = GraphModelConfig(
    n_components=12,
    alpha=0.02,
    alpha_switch=0.02,
    allow_abundance_abundance=True,
    alpha_abundance_grid=_ALPHA_ABUNDANCE_GRID_GRAPH_LATENT,
    min_module_size=2,
)
_NOISY_GRAPH_CONFIG = GraphModelConfig(
    n_components=8,
    alpha=0.02,
    alpha_switch=0.02,
    allow_abundance_abundance=True,
    alpha_abundance_grid=_ALPHA_ABUNDANCE_GRID_GRAPH_LATENT,
    min_module_size=2,
)


# ---------------------------------------------------------------------------
# Session fixture
# ---------------------------------------------------------------------------

from isograph.benchmarks.synthetic import generate_multiplex_suite
from isograph.evaluation.metrics import (
    giant_component_fraction,
    module_recovery_score,
    role_aware_recall,
)
from isograph.io.artifacts import load_dataset_bundle


@pytest.fixture(scope="session")
def multiplex_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("multiplex_datasets")
    paths = generate_multiplex_suite(root, seed=7)
    return {p.name: p for p in paths}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fit_vae(dataset_dir: Path, config: VaeModelConfig):
    from isograph.models.vae import VaeNetworkModel

    bundle = load_dataset_bundle(dataset_dir)
    model = VaeNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
        gene_counts=bundle.matrices.get("gene_counts"),
        gene_table=bundle.feature_tables.get("gene"),
    )
    return artifacts, bundle


def _fit_graph(dataset_dir: Path, config: GraphModelConfig):
    from isograph.models.graph import GraphNetworkModel

    bundle = load_dataset_bundle(dataset_dir)
    model = GraphNetworkModel(config)
    artifacts = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table,
        gene_counts=bundle.matrices.get("gene_counts"),
        gene_table=bundle.feature_tables.get("gene"),
    )
    return artifacts, bundle


# ---------------------------------------------------------------------------
# Fast schema / fixture tests (no slow mark)
# ---------------------------------------------------------------------------


def test_multiplex_suite_fixture_names(multiplex_suite):
    expected = {
        "toy_multiplex_v1",
        "medium_multiplex_v1",
        "noisy_multiplex_v1",
        "large_multiplex_v1",
    }
    assert expected == set(multiplex_suite.keys())


def test_multiplex_fixtures_have_truth_tables(multiplex_suite):
    for name, path in multiplex_suite.items():
        bundle = load_dataset_bundle(path)
        assert "truth_modules.parquet" in bundle.truth_tables, f"{name}: missing truth_modules"
        assert (
            "truth_channel_role.parquet" in bundle.truth_tables
        ), f"{name}: missing truth_channel_role"


def test_multiplex_fixtures_have_gene_counts(multiplex_suite):
    for name, path in multiplex_suite.items():
        bundle = load_dataset_bundle(path)
        assert "gene_counts" in bundle.matrices, f"{name}: missing gene_counts matrix"


def test_truth_channel_role_schema(multiplex_suite):
    bundle = load_dataset_bundle(multiplex_suite["medium_multiplex_v1"])
    truth_role = bundle.truth_tables["truth_channel_role.parquet"]
    for col in ("gene_id", "module_id", "truth_role"):
        assert col in truth_role.columns, f"Missing column: {col}"


def test_role_aware_recall_returns_all_roles(multiplex_suite):
    bundle = load_dataset_bundle(multiplex_suite["toy_multiplex_v1"])
    truth_modules = bundle.truth_tables["truth_modules.parquet"]
    truth_role = bundle.truth_tables["truth_channel_role.parquet"]
    result = role_aware_recall(truth_modules, truth_role)
    for role in ("switch_only", "abundance_only", "coupled", "discordant"):
        assert role in result, f"Missing role: {role}"
        assert 0.0 <= result[role] <= 1.0, f"Out-of-range recall for {role}"


def test_giant_component_fraction_empty_edge_table():
    assert giant_component_fraction(pd.DataFrame(columns=["source", "target"]), 100) == 0.0


def test_giant_component_fraction_zero_genes():
    edge_table = pd.DataFrame({"source": ["A"], "target": ["B"]})
    assert giant_component_fraction(edge_table, 0) == 0.0


# ---------------------------------------------------------------------------
# Slow gate tests — VAE
# ---------------------------------------------------------------------------


@_skip_no_torch
@pytest.mark.slow
def test_medium_multiplex_vae_no_giant_component(multiplex_suite):
    arts, bundle = _fit_vae(multiplex_suite["medium_multiplex_v1"], _MEDIUM_VAE_CONFIG)
    n_genes = len(bundle.feature_tables.get("gene", pd.DataFrame()))
    frac = giant_component_fraction(arts.edge_table, n_genes)
    assert frac < 0.50, f"Giant component fraction {frac:.3f} >= 0.50 on medium_multiplex_v1"


@_skip_no_torch
@pytest.mark.slow
def test_noisy_multiplex_vae_no_giant_component(multiplex_suite):
    arts, bundle = _fit_vae(multiplex_suite["noisy_multiplex_v1"], _NOISY_VAE_CONFIG)
    n_genes = len(bundle.feature_tables.get("gene", pd.DataFrame()))
    frac = giant_component_fraction(arts.edge_table, n_genes)
    assert frac < 0.50, f"Giant component fraction {frac:.3f} >= 0.50 on noisy_multiplex_v1"


@_skip_no_torch
@pytest.mark.slow
def test_large_multiplex_vae_no_giant_component(multiplex_suite):
    arts, bundle = _fit_vae(multiplex_suite["large_multiplex_v1"], _LARGE_VAE_CONFIG)
    n_genes = len(bundle.feature_tables.get("gene", pd.DataFrame()))
    frac = giant_component_fraction(arts.edge_table, n_genes)
    assert frac < 0.50, f"Giant component fraction {frac:.3f} >= 0.50 on large_multiplex_v1"


@_skip_no_torch
@pytest.mark.slow
def test_toy_multiplex_vae_recovery(multiplex_suite):
    if _GATE_TOY_VAE is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_vae(multiplex_suite["toy_multiplex_v1"], _TOY_VAE_CONFIG)
    truth = bundle.truth_tables["truth_modules.parquet"]
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_TOY_VAE, f"recovery={recovery:.4f} < gate={_GATE_TOY_VAE}"


@_skip_no_torch
@pytest.mark.slow
def test_medium_multiplex_vae_recovery(multiplex_suite):
    if _GATE_MEDIUM_VAE is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_vae(multiplex_suite["medium_multiplex_v1"], _MEDIUM_VAE_CONFIG)
    truth = bundle.truth_tables["truth_modules.parquet"]
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_MEDIUM_VAE, f"recovery={recovery:.4f} < gate={_GATE_MEDIUM_VAE}"


@_skip_no_torch
@pytest.mark.slow
def test_noisy_multiplex_vae_recovery(multiplex_suite):
    if _GATE_NOISY_VAE is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_vae(multiplex_suite["noisy_multiplex_v1"], _NOISY_VAE_CONFIG)
    truth = bundle.truth_tables["truth_modules.parquet"]
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_NOISY_VAE, f"recovery={recovery:.4f} < gate={_GATE_NOISY_VAE}"


@_skip_no_torch
@pytest.mark.slow
def test_large_multiplex_vae_recovery(multiplex_suite):
    if _GATE_LARGE_VAE is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_vae(multiplex_suite["large_multiplex_v1"], _LARGE_VAE_CONFIG)
    truth = bundle.truth_tables["truth_modules.parquet"]
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_LARGE_VAE, f"recovery={recovery:.4f} < gate={_GATE_LARGE_VAE}"


@_skip_no_torch
@pytest.mark.slow
def test_medium_multiplex_vae_abundance_recall(multiplex_suite):
    if _GATE_AB_RECALL_MEDIUM_VAE is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_vae(multiplex_suite["medium_multiplex_v1"], _MEDIUM_VAE_CONFIG)
    truth_role = bundle.truth_tables["truth_channel_role.parquet"]
    recall = role_aware_recall(arts.module_table, truth_role)["abundance_only"]
    assert (
        recall >= _GATE_AB_RECALL_MEDIUM_VAE
    ), f"abundance_only recall={recall:.4f} < gate={_GATE_AB_RECALL_MEDIUM_VAE}"


@_skip_no_torch
@pytest.mark.slow
def test_noisy_multiplex_vae_abundance_recall(multiplex_suite):
    if _GATE_AB_RECALL_NOISY_VAE is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_vae(multiplex_suite["noisy_multiplex_v1"], _NOISY_VAE_CONFIG)
    truth_role = bundle.truth_tables["truth_channel_role.parquet"]
    recall = role_aware_recall(arts.module_table, truth_role)["abundance_only"]
    assert (
        recall >= _GATE_AB_RECALL_NOISY_VAE
    ), f"abundance_only recall={recall:.4f} < gate={_GATE_AB_RECALL_NOISY_VAE}"


@_skip_no_torch
@pytest.mark.slow
def test_large_multiplex_vae_abundance_recall(multiplex_suite):
    if _GATE_AB_RECALL_LARGE_VAE is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_vae(multiplex_suite["large_multiplex_v1"], _LARGE_VAE_CONFIG)
    truth_role = bundle.truth_tables["truth_channel_role.parquet"]
    recall = role_aware_recall(arts.module_table, truth_role)["abundance_only"]
    assert (
        recall >= _GATE_AB_RECALL_LARGE_VAE
    ), f"abundance_only recall={recall:.4f} < gate={_GATE_AB_RECALL_LARGE_VAE}"


# ---------------------------------------------------------------------------
# Slow gate tests — graph
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_medium_multiplex_graph_no_giant_component(multiplex_suite):
    arts, bundle = _fit_graph(multiplex_suite["medium_multiplex_v1"], _MEDIUM_GRAPH_CONFIG)
    n_genes = len(bundle.feature_tables.get("gene", pd.DataFrame()))
    frac = giant_component_fraction(arts.edge_table, n_genes)
    assert (
        frac < 0.50
    ), f"Giant component fraction {frac:.3f} >= 0.50 on medium_multiplex_v1 (graph)"


@pytest.mark.slow
def test_noisy_multiplex_graph_no_giant_component(multiplex_suite):
    arts, bundle = _fit_graph(multiplex_suite["noisy_multiplex_v1"], _NOISY_GRAPH_CONFIG)
    n_genes = len(bundle.feature_tables.get("gene", pd.DataFrame()))
    frac = giant_component_fraction(arts.edge_table, n_genes)
    assert frac < 0.50, f"Giant component fraction {frac:.3f} >= 0.50 on noisy_multiplex_v1 (graph)"


@pytest.mark.slow
def test_medium_multiplex_graph_recovery(multiplex_suite):
    if _GATE_MEDIUM_GRAPH is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_graph(multiplex_suite["medium_multiplex_v1"], _MEDIUM_GRAPH_CONFIG)
    truth = bundle.truth_tables["truth_modules.parquet"]
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_MEDIUM_GRAPH, f"recovery={recovery:.4f} < gate={_GATE_MEDIUM_GRAPH}"


@pytest.mark.slow
def test_noisy_multiplex_graph_recovery(multiplex_suite):
    if _GATE_NOISY_GRAPH is None:
        pytest.skip("Gate not set — run calibration benchmark first")
    arts, bundle = _fit_graph(multiplex_suite["noisy_multiplex_v1"], _NOISY_GRAPH_CONFIG)
    truth = bundle.truth_tables["truth_modules.parquet"]
    recovery = module_recovery_score(arts.module_table, truth)
    assert recovery >= _GATE_NOISY_GRAPH, f"recovery={recovery:.4f} < gate={_GATE_NOISY_GRAPH}"
