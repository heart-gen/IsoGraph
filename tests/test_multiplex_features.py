from __future__ import annotations

import numpy as np
import pandas as pd

from isograph.features.channels import gene_feature_channels
from isograph.io.real_data import filter_by_expr
from isograph.models.base import compute_module_gene_roles


def test_filter_by_expr_design_uses_hat_min_sample_size() -> None:
    counts = np.array(
        [
            [20, 20, 20, 20],
            [20, 20, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )
    design = np.ones((4, 1), dtype=float)
    keep = filter_by_expr(counts, design=design, lib_size=np.repeat(1_000_000.0, 4))
    assert keep.tolist() == [True, False, False]


def test_gene_feature_channels_keep_single_transcript_abundance_only() -> None:
    transcript_table = pd.DataFrame(
        {
            "transcript_id": ["A_T1", "A_T2", "B_T1"],
            "gene_id": ["GeneA", "GeneA", "GeneB"],
        }
    )
    counts = np.array([[10, 20, 30], [30, 20, 10], [5, 7, 9]], dtype=float)
    matrix, info = gene_feature_channels(counts, transcript_table)
    assert matrix.shape[0] == 3
    assert set(info.loc[info["gene_id"] == "GeneB", "feature_type"]) == {"abundance"}
    assert set(info.loc[info["gene_id"] == "GeneA", "feature_type"]) == {"abundance", "switch"}


def test_module_gene_roles_classify_coupled_and_discordant() -> None:
    samples = ["S1", "S2", "S3", "S4"]
    feature_scores = pd.DataFrame(
        {
            "feature_id": [
                "G1::abundance",
                "G1::switch",
                "G2::abundance",
                "G2::switch",
                "G3::abundance",
            ],
            "gene_id": ["G1", "G1", "G2", "G2", "G3"],
            "feature_type": ["abundance", "switch", "abundance", "switch", "abundance"],
            "S1": [1, 1, 1, -1, 1],
            "S2": [2, 2, 2, -2, 2],
            "S3": [3, 3, 3, -3, 3],
            "S4": [4, 4, 4, -4, 4],
        }
    )
    module_table = pd.DataFrame({"module_id": ["M000", "M000", "M000"], "gene_id": ["G1", "G2", "G3"]})
    roles = compute_module_gene_roles(module_table, feature_scores, pd.DataFrame({"sample_id": samples}), min_abs_r=0.1)
    role_map = roles.set_index("gene_id")["module_role"].to_dict()
    assert role_map["G1"] == "coupled"
    assert role_map["G2"] == "discordant"
    assert role_map["G3"] == "abundance_only"


def test_node_diagnostics_classifies_gene_fates() -> None:
    """node_stats + _build_node_diagnostics explain why a gene is absent from modules.

    Covers all four fates: assigned (a switch clique), sub_min_module_size (a gene
    whose partner is downweighted away leaving a too-small community), the
    downweighted partner itself (downweighted_below_alpha), and a gene whose best
    association is below the alpha pre-filter (isolated_below_alpha, NaN max-assoc —
    the DRD2 case).
    """
    from isograph.models.multiplex import project_feature_similarity_to_gene_graph
    from isograph.models.vae import _build_node_diagnostics

    genes = [f"g{i}" for i in range(6)]
    feature_info = pd.DataFrame({
        "feature_id": [f"{g}::switch" for g in genes],
        "gene_id": genes,
        "feature_type": ["switch"] * 6,
    })
    n = 6
    sim = np.eye(n)
    def _set(a, b, v):
        sim[a, b] = sim[b, a] = v
    _set(0, 1, 0.9); _set(0, 2, 0.85); _set(1, 2, 0.8)   # clique -> module
    _set(3, 4, 0.7)                                       # g4 downweighted away
    _set(5, 0, 0.3)                                       # g5 below alpha (DRD2-like)

    alpha = 0.5
    reliability = {g: 1.0 for g in genes}
    reliability["g4"] = 0.3   # 0.7 * sqrt(1*0.3) = 0.383 < alpha -> edge dropped

    node_stats: dict = {}
    _, edge_rows = project_feature_similarity_to_gene_graph(
        sim, feature_info, alpha, alpha_switch=alpha,
        gene_reliability=reliability, node_stats=node_stats,
    )
    module_table = pd.DataFrame({"gene_id": ["g0", "g1", "g2"], "module_id": ["M000"] * 3})
    diag = _build_node_diagnostics(
        feature_info, edge_rows, module_table, reliability, node_stats,
        alpha_switch=alpha, min_module_size=3, reliability_on=True,
    ).set_index("gene_id")

    assert diag.loc["g0", "fate"] == "assigned"
    assert diag.loc["g0", "module_id"] == "M000"
    assert diag.loc["g4", "fate"] == "downweighted_below_alpha"
    assert diag.loc["g4", "switch_reliability"] == 0.3
    # g5's strongest association (0.3) is below the alpha pre-filter -> NaN max-assoc.
    assert diag.loc["g5", "fate"] == "isolated_below_alpha"
    assert np.isnan(diag.loc["g5", "max_abs_assoc"])
    assert diag.loc["g5", "degree"] == 0
