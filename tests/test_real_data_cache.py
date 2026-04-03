from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from isograph.io.artifacts import load_dataset_bundle
from isograph.io.real_data import (
    _gene_bucket,
    _selection_key,
    _transcript_cache_metadata_path,
    freeze_real_dataset,
)
from isograph.workflow.config import RealDataFreezeConfig


def _write_tsv(path: Path, frame: pd.DataFrame, gzip_output: bool = False) -> None:
    if gzip_output:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            frame.to_csv(handle, sep="\t", index=False)
        return
    frame.to_csv(path, sep="\t", index=False)


def test_gene_bucket_is_stable() -> None:
    assert _gene_bucket("ENSG000001") == _gene_bucket("ENSG000001")
    assert 0 <= _gene_bucket("ENSG000001") < 128
    assert 0 <= _gene_bucket("ENSG000002") < 128


def test_freeze_real_dataset_builds_repo_local_transcript_cache(tmp_path: Path) -> None:
    counts_root = tmp_path / "counts"
    annotations_root = tmp_path / "annotations"
    cache_root = tmp_path / "cache"
    suite_dir = tmp_path / "datasets" / "stage0"
    counts_root.mkdir()
    annotations_root.mkdir()

    phenotype = pd.DataFrame(
        {
            "Region": ["Caudate", "Caudate"],
            "Race": ["AA", "AA"],
            "Age": [45, 52],
            "dropped": ["f", "f"],
            "Dx": ["Control", "SCZD"],
            "RNum": ["R1", "R2"],
            "BrNum": ["B1", "B2"],
        }
    )
    ancestry = pd.DataFrame({"id": ["B1", "B2"], "AFR": [0.95, 0.93]})
    gene_counts = pd.DataFrame(
        {
            "Geneid": ["G1", "G2"],
            "Chr": ["chr1", "chr2"],
            "Start": [10, 20],
            "End": [11, 21],
            "Strand": ["+", "-"],
            "Length": [1000, 900],
            "R1": [100.0, 20.0],
            "R2": [120.0, 25.0],
        }
    )
    tx_counts = pd.DataFrame(
        {
            "Name": ["T1", "T2", "T3"],
            "Length": [500, 520, 450],
            "EffectiveLength": [450.0, 470.0, 400.0],
            "R1": [60.0, 40.0, 20.0],
            "R2": [70.0, 50.0, 25.0],
        }
    )
    tx_annot = pd.DataFrame(
        {
            "transcript_id": ["T1", "T2", "T3"],
            "gene_id": ["G1", "G1", "G2"],
        }
    )
    psi_counts = pd.DataFrame(
        {
            "gene_id": ["G1", "G2"],
            "event_type": ["SE", "A5SS"],
            "event_info": ["event1", "event2"],
            "chr": ["chr1", "chr2"],
            "strand": ["+", "-"],
            "R1": [0.25, 0.70],
            "R2": [0.35, 0.60],
        }
    )
    psi_annot = pd.DataFrame(
        {
            "gene_id": ["G1", "G2"],
            "event_type": ["SE", "A5SS"],
            "event_info": ["event1", "event2"],
            "chr": ["chr1", "chr2"],
            "strand": ["+", "-"],
            "psi_uid": ["P1", "P2"],
        }
    )

    _write_tsv(tmp_path / "phenotypes.tsv", phenotype)
    ancestry.to_csv(tmp_path / "ancestry.txt", sep=" ", index=False)
    _write_tsv(counts_root / "gene-counts.tsv", gene_counts)
    _write_tsv(counts_root / "tx-counts.tsv", tx_counts)
    _write_tsv(counts_root / "psi-events.tsv.gz", psi_counts, gzip_output=True)
    _write_tsv(annotations_root / "transcript_annotation.tsv.gz", tx_annot, gzip_output=True)
    _write_tsv(annotations_root / "psi_annotation.tsv.gz", psi_annot, gzip_output=True)

    config = RealDataFreezeConfig(
        counts_root=counts_root,
        annotations_root=annotations_root,
        phenotype_tsv=tmp_path / "phenotypes.tsv",
        ancestry_tsv=tmp_path / "ancestry.txt",
        output_name="realmini_v1",
        gene_panel_size=2,
        cache_root=cache_root,
    )

    dataset_dir = freeze_real_dataset(config, suite_dir)
    bundle = load_dataset_bundle(dataset_dir)
    assert sorted(bundle.feature_tables) == ["gene", "psi", "transcript"]
    assert bundle.matrices["transcript_counts"].shape == (3, 2)
    assert sorted(bundle.feature_tables["transcript"]["transcript_id"]) == ["T1", "T2", "T3"]

    source_cache_dir = cache_root / "sources" / _selection_key(config)
    metadata_path = _transcript_cache_metadata_path(source_cache_dir)
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["bucket_count"] == 128
    assert metadata["row_count"] == 3
    assert sorted(metadata["selected_samples"]) == ["R1", "R2"]
    assert any(path.is_dir() for path in (source_cache_dir / "transcript_counts").glob("gene_bucket=*"))

    cached_dataset_dir = freeze_real_dataset(config, suite_dir)
    assert cached_dataset_dir == dataset_dir
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    assert manifest["provenance"]["splicing_feature_type"] == "psi"
    assert "transcript_cache_metadata" in manifest["provenance"]
