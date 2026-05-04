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
