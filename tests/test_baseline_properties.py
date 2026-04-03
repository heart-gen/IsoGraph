from __future__ import annotations

import numpy as np
import pandas as pd

from isograph.io.artifacts import load_dataset_bundle
from isograph.models.baseline import BaselineNetworkModel
from isograph.workflow.config import BaselineModelConfig


def test_sample_permutation_keeps_module_assignments(synthetic_suite) -> None:
    bundle = load_dataset_bundle(synthetic_suite / "toy_v1")
    model = BaselineNetworkModel(BaselineModelConfig(alpha=0.05))
    original = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table.rename(columns={"sample_id": "RNum"}),
    )

    perm = np.arange(len(bundle.sample_table))[::-1]
    permuted = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"][:, perm],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table.rename(columns={"sample_id": "RNum"}).iloc[perm].reset_index(drop=True),
    )

    left = original.module_table.sort_values(["gene_id", "module_id"]).reset_index(drop=True)
    right = permuted.module_table.sort_values(["gene_id", "module_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_transcript_relabeling_preserves_gene_assignments(synthetic_suite) -> None:
    bundle = load_dataset_bundle(synthetic_suite / "toy_v1")
    model = BaselineNetworkModel(BaselineModelConfig(alpha=0.05))
    original = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=bundle.feature_tables["transcript"],
        sample_table=bundle.sample_table.rename(columns={"sample_id": "RNum"}),
    )
    transcript_table = bundle.feature_tables["transcript"].copy()
    transcript_table["transcript_id"] = transcript_table["transcript_id"].sample(
        frac=1.0, random_state=11
    ).to_numpy()
    result = model.fit(
        transcript_counts=bundle.matrices["transcript_counts"],
        transcript_table=transcript_table,
        sample_table=bundle.sample_table.rename(columns={"sample_id": "RNum"}),
    )
    assert set(result.module_table["gene_id"]) == set(original.module_table["gene_id"])
