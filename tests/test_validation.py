from __future__ import annotations

import pytest
from pydantic import ValidationError

import numpy as np
import pandas as pd

from isograph.io.artifacts import DatasetBundle, load_dataset_bundle, save_dataset_bundle
from isograph.validation import DatasetManifest
from isograph.io.artifacts import build_feature_spec, build_matrix_spec


def test_manifest_rejects_missing_feature_tables() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest(
            dataset_name="toy_v1",
            suite_name="core_v1",
            description="bad",
            sample_table="samples.parquet",
            feature_tables=[],
            matrices=[],
            provenance={},
        )


def test_dataset_bundle_without_junctions_round_trips(tmp_path) -> None:
    sample_table = pd.DataFrame({"sample_id": ["S1", "S2"]})
    gene_table = pd.DataFrame({"gene_id": ["G1"]})
    tx_table = pd.DataFrame({"transcript_id": ["T1"], "gene_id": ["G1"]})
    psi_table = pd.DataFrame({"psi_uid": ["P1"], "gene_id": ["G1"]})
    gene_counts = np.array([[1.0, 2.0]])
    tx_counts = np.array([[0.5, 1.5]])
    psi = np.array([[0.2, 0.8]])
    manifest = DatasetManifest(
        dataset_name="realmini_v1",
        suite_name="stage0",
        description="psi-only realmini",
        sample_table="samples.parquet",
        feature_tables=[
            build_feature_spec("gene", "genes.parquet", gene_table),
            build_feature_spec("transcript", "transcripts.parquet", tx_table),
            build_feature_spec("psi", "psi.parquet", psi_table),
        ],
        matrices=[
            build_matrix_spec("gene_counts", "gene_counts.npz", gene_counts),
            build_matrix_spec("transcript_counts", "transcript_counts.npz", tx_counts),
            build_matrix_spec("psi", "psi.npz", psi),
        ],
        provenance={"splicing_feature_type": "psi"},
    )
    bundle = DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables={"gene": gene_table, "transcript": tx_table, "psi": psi_table},
        matrices={"gene_counts": gene_counts, "transcript_counts": tx_counts, "psi": psi},
        truth_tables={},
    )
    dataset_dir = save_dataset_bundle(bundle, tmp_path / "realmini_v1")
    loaded = load_dataset_bundle(dataset_dir)
    assert sorted(loaded.feature_tables) == ["gene", "psi", "transcript"]
    assert sorted(loaded.matrices) == ["gene_counts", "psi", "transcript_counts"]
