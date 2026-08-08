"""Stage 8C gate tests — structural annotation from GTF."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.explain.annotation import (
    _ALL_ANNOTATION_COLUMNS,
    _BOOLEAN_ANNOTATION_COLUMNS,
    annotate_driver_table,
    annotate_transcript_table,
    load_annotation_table,
)
from isograph.explain.structure import (
    STRUCTURAL_LABELS,
    TranscriptRecord,
    annotate_switch_pairs,
    compare_pair,
    parse_gtf,
)

# ---------------------------------------------------------------------------
# GTF fixture helpers
# ---------------------------------------------------------------------------


def _gtf_line(
    chrom: str,
    feature: str,
    start: int,
    end: int,
    strand: str,
    gene_id: str,
    transcript_id: str = "",
    biotype: str = "protein_coding",
    extra: str = "",
) -> str:
    attrs = f'gene_id "{gene_id}";'
    if transcript_id:
        attrs += f' transcript_id "{transcript_id}"; transcript_biotype "{biotype}";'
    if extra:
        attrs += f" {extra}"
    return f"{chrom}\tHAVANA\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}\n"


def _write_gtf(path: Path, lines: list[str]) -> None:
    with open(path, "w") as fh:
        fh.writelines(lines)


def _write_gtf_gz(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt") as fh:
        fh.writelines(lines)


def _simple_gtf_lines() -> list[str]:
    """Gene G1 with two transcripts: T1 (3 exons, CDS) and T2 (2 exons, no CDS)."""
    return [
        _gtf_line("chr1", "gene", 1000, 5000, "+", "G1"),
        _gtf_line("chr1", "transcript", 1000, 5000, "+", "G1", "T1", "protein_coding"),
        _gtf_line("chr1", "exon", 1000, 1500, "+", "G1", "T1"),
        _gtf_line("chr1", "exon", 2000, 2500, "+", "G1", "T1"),
        _gtf_line("chr1", "exon", 4000, 5000, "+", "G1", "T1"),
        _gtf_line("chr1", "CDS", 1100, 1500, "+", "G1", "T1"),
        _gtf_line("chr1", "CDS", 2000, 2500, "+", "G1", "T1"),
        _gtf_line("chr1", "CDS", 4000, 4800, "+", "G1", "T1"),
        _gtf_line("chr1", "transcript", 1000, 5000, "+", "G1", "T2", "retained_intron"),
        _gtf_line("chr1", "exon", 1200, 1500, "+", "G1", "T2"),
        _gtf_line("chr1", "exon", 4000, 5000, "+", "G1", "T2"),
    ]


def _make_switch_pairs_df(gene_id: str, t1: str, t2: str) -> pd.DataFrame:
    return pd.DataFrame({"gene_id": [gene_id], "transcript_id_1": [t1], "transcript_id_2": [t2]})


# ---------------------------------------------------------------------------
# ExplainResult fixture helpers (reuse 8A pattern)
# ---------------------------------------------------------------------------


def _make_synthetic_explain_inputs(
    n_genes: int = 24,
    n_samples: int = 48,
    n_modules: int = 3,
    n_transcripts_per_gene: int = 2,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    gene_ids = [f"G{i:04d}" for i in range(n_genes)]
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    genes_per_module = n_genes // n_modules
    module_assignment = np.repeat(np.arange(n_modules), genes_per_module)[:n_genes]
    module_latent = rng.normal(size=(n_modules, n_samples))
    switch_coords = np.vstack(
        [
            module_latent[module_assignment[i]] + rng.normal(0, 0.3, n_samples)
            for i in range(n_genes)
        ]
    )
    feature_scores = pd.DataFrame(switch_coords, columns=sample_ids)
    feature_scores.insert(0, "gene_id", gene_ids)
    modules = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "module_id": [f"M{m:03d}" for m in module_assignment],
        }
    )
    transcript_ids = [
        f"G{i:04d}_T{t}" for i in range(n_genes) for t in range(n_transcripts_per_gene)
    ]
    tx_data = np.zeros((n_samples, len(transcript_ids)))
    for i in range(n_genes):
        signal = module_latent[module_assignment[i]] * 0.2
        p = np.clip(0.5 + signal * 0.1, 0.05, 0.95)
        col0 = i * n_transcripts_per_gene
        tx_data[:, col0] = np.clip(p + rng.normal(0, 0.05, n_samples), 0.01, 0.99)
        tx_data[:, col0 + 1] = np.clip(
            1.0 - tx_data[:, col0] + rng.normal(0, 0.05, n_samples), 0.01, 0.99
        )
    feature_table = pd.DataFrame(tx_data, index=sample_ids, columns=transcript_ids)
    feature_meta = pd.DataFrame(
        [
            {
                "feature_id": f"G{i:04d}_T{t}",
                "gene_id": f"G{i:04d}",
                "transcript_id": f"G{i:04d}_T{t}",
                "feature_type": "transcript_usage",
            }
            for i in range(n_genes)
            for t in range(n_transcripts_per_gene)
        ]
    )
    return modules, feature_scores, feature_table, feature_meta, sample_ids


def _write_artifact(tmp_path: Path, modules: pd.DataFrame, feature_scores: pd.DataFrame) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    modules.to_parquet(artifact_dir / "modules.parquet", index=False)
    feature_scores.to_parquet(artifact_dir / "feature_scores.parquet", index=False)
    return artifact_dir


def _make_annotation_df(gene_ids: list[str], transcript_ids: list[str]) -> pd.DataFrame:
    """Build a minimal annotation DataFrame for the given transcripts."""
    len(transcript_ids)
    rng = np.random.default_rng(42)
    row_list = []
    for tx_id, gene_id in zip(transcript_ids, gene_ids, strict=False):
        row: dict = {"feature_id": tx_id, "gene_id": gene_id}
        for col in _BOOLEAN_ANNOTATION_COLUMNS:
            row[col] = bool(rng.integers(0, 2))
        row["transcript_length_delta"] = float(rng.integers(0, 500))
        row["cds_length_delta"] = float(rng.integers(0, 300))
        row["shared_exon_fraction"] = float(rng.uniform(0.3, 1.0))
        row_list.append(row)
    df = pd.DataFrame(row_list)
    for col in _BOOLEAN_ANNOTATION_COLUMNS:
        df[col] = pd.array(df[col].tolist(), dtype=pd.BooleanDtype())
    return df


# ===========================================================================
# Group 1: parse_gtf
# ===========================================================================


def test_parse_gtf_basic(tmp_path):
    gtf_path = tmp_path / "test.gtf"
    _write_gtf(gtf_path, _simple_gtf_lines())
    db = parse_gtf(gtf_path)
    assert "T1" in db
    assert "T2" in db
    assert db["T1"].gene_id == "G1"
    assert len(db["T1"].exons) == 3
    assert len(db["T2"].exons) == 2


def test_parse_gtf_gzip(tmp_path):
    gtf_path = tmp_path / "test.gtf.gz"
    _write_gtf_gz(gtf_path, _simple_gtf_lines())
    db = parse_gtf(gtf_path)
    assert "T1" in db and "T2" in db
    assert len(db["T1"].exons) == 3


def test_parse_gtf_cds(tmp_path):
    gtf_path = tmp_path / "test.gtf"
    _write_gtf(gtf_path, _simple_gtf_lines())
    db = parse_gtf(gtf_path)
    assert len(db["T1"].cds) == 3
    assert len(db["T2"].cds) == 0


def test_parse_gtf_biotype(tmp_path):
    gtf_path = tmp_path / "test.gtf"
    _write_gtf(gtf_path, _simple_gtf_lines())
    db = parse_gtf(gtf_path)
    assert db["T1"].biotype == "protein_coding"
    assert db["T2"].biotype == "retained_intron"


# ===========================================================================
# Group 2: compare_pair
# ===========================================================================


def _make_tx(
    tx_id: str,
    gene_id: str,
    exons: list[tuple[int, int]],
    cds: list[tuple[int, int]] | None = None,
    biotype: str = "protein_coding",
    strand: str = "+",
) -> TranscriptRecord:
    return TranscriptRecord(
        transcript_id=tx_id,
        gene_id=gene_id,
        chrom="chr1",
        strand=strand,
        biotype=biotype,
        exons=sorted(exons),
        cds=sorted(cds or []),
    )


def test_compare_pair_first_exon_changed():
    tx1 = _make_tx("T1", "G1", [(1000, 1500), (2000, 2500), (4000, 5000)])
    tx2 = _make_tx("T2", "G1", [(800, 1500), (2000, 2500), (4000, 5000)])
    result = compare_pair(tx1, tx2)
    assert result["first_exon_changed"] is True
    assert result["last_exon_changed"] is False


def test_compare_pair_last_exon_changed():
    tx1 = _make_tx("T1", "G1", [(1000, 1500), (2000, 2500), (4000, 5000)])
    tx2 = _make_tx("T2", "G1", [(1000, 1500), (2000, 2500), (4000, 5200)])
    result = compare_pair(tx1, tx2)
    assert result["last_exon_changed"] is True
    assert result["first_exon_changed"] is False


def test_compare_pair_internal_exon_difference():
    tx1 = _make_tx("T1", "G1", [(1000, 1500), (2000, 2500), (3000, 3200), (4000, 5000)])
    tx2 = _make_tx("T2", "G1", [(1000, 1500), (4000, 5000)])  # skips internal exons
    result = compare_pair(tx1, tx2)
    assert result["internal_exon_difference"] is True


def test_compare_pair_coding_status_change():
    tx1 = _make_tx("T1", "G1", [(1000, 1500), (2000, 2500)], cds=[(1100, 1500), (2000, 2400)])
    tx2 = _make_tx("T2", "G1", [(1000, 1500), (2000, 2500)], cds=[], biotype="retained_intron")
    result = compare_pair(tx1, tx2)
    assert result["coding_status_change"] is True
    assert result["cds_length_delta"] > 0


def test_compare_pair_identical_transcripts():
    tx1 = _make_tx("T1", "G1", [(1000, 1500), (2000, 2500)], cds=[(1100, 1400)])
    tx2 = _make_tx("T1", "G1", [(1000, 1500), (2000, 2500)], cds=[(1100, 1400)])
    result = compare_pair(tx1, tx2)
    assert result["first_exon_changed"] is False
    assert result["last_exon_changed"] is False
    assert result["internal_exon_difference"] is False
    assert result["cds_changed"] is False
    assert result["coding_status_change"] is False
    assert result["transcript_length_delta"] == 0
    assert result["cds_length_delta"] == 0
    assert result["shared_exon_fraction"] == pytest.approx(1.0)


def test_compare_pair_single_exon_transcripts():
    tx1 = _make_tx("T1", "G1", [(1000, 2000)])
    tx2 = _make_tx("T2", "G1", [(1000, 2000)])
    result = compare_pair(tx1, tx2)
    assert result["internal_exon_difference"] is False


# ===========================================================================
# Group 3: annotate_switch_pairs
# ===========================================================================


def test_annotate_switch_pairs_output_schema(tmp_path):
    gtf_path = tmp_path / "test.gtf"
    _write_gtf(gtf_path, _simple_gtf_lines())
    pairs = _make_switch_pairs_df("G1", "T1", "T2")
    result = annotate_switch_pairs(pairs, gtf_path)
    assert set(result.columns) >= {"transcript_id", "gene_id"} | set(STRUCTURAL_LABELS)
    assert set(result["transcript_id"]) == {"T1", "T2"}


def test_annotate_switch_pairs_per_transcript_aggregation(tmp_path):
    """Transcript that differs in first exon across any pair → first_exon_changed=True."""
    lines = _simple_gtf_lines()
    # Add T3: same first exon as T1 but different last exon
    lines += [
        _gtf_line("chr1", "transcript", 1000, 6000, "+", "G1", "T3", "protein_coding"),
        _gtf_line("chr1", "exon", 1000, 1500, "+", "G1", "T3"),
        _gtf_line("chr1", "exon", 4000, 6000, "+", "G1", "T3"),
    ]
    gtf_path = tmp_path / "test.gtf"
    _write_gtf(gtf_path, lines)
    # T1 vs T2: first exon differs; T1 vs T3: first exon same
    pairs = pd.DataFrame(
        {
            "gene_id": ["G1", "G1"],
            "transcript_id_1": ["T1", "T1"],
            "transcript_id_2": ["T2", "T3"],
        }
    )
    result = annotate_switch_pairs(pairs, gtf_path)
    t1_row = result[result["transcript_id"] == "T1"].iloc[0]
    assert t1_row["first_exon_changed"] == True  # noqa: E712 — any() across pairs


def test_annotate_switch_pairs_missing_transcript_warns(tmp_path):
    gtf_path = tmp_path / "test.gtf"
    _write_gtf(gtf_path, _simple_gtf_lines())
    pairs = pd.DataFrame(
        {
            "gene_id": ["G1"],
            "transcript_id_1": ["T1"],
            "transcript_id_2": ["TX_MISSING"],
        }
    )
    with pytest.warns(UserWarning, match="not found in GTF"):
        result = annotate_switch_pairs(pairs, gtf_path)
    # T1 should still appear (with pd.NA labels since partner missing)
    assert "T1" in result["transcript_id"].values


def test_annotate_switch_pairs_missing_columns_raises(tmp_path):
    gtf_path = tmp_path / "test.gtf"
    _write_gtf(gtf_path, _simple_gtf_lines())
    bad_pairs = pd.DataFrame({"transcript_id_1": ["T1"], "transcript_id_2": ["T2"]})
    with pytest.raises(ValueError, match="missing required columns"):
        annotate_switch_pairs(bad_pairs, gtf_path)


def test_annotate_switch_pairs_boolean_dtype(tmp_path):
    gtf_path = tmp_path / "test.gtf"
    _write_gtf(gtf_path, _simple_gtf_lines())
    pairs = _make_switch_pairs_df("G1", "T1", "T2")
    result = annotate_switch_pairs(pairs, gtf_path)
    for col in [
        "first_exon_changed",
        "last_exon_changed",
        "cds_changed",
        "coding_status_change",
        "biotype_switch",
    ]:
        assert isinstance(
            result[col].dtype, pd.BooleanDtype
        ), f"Expected BooleanDtype for {col}, got {result[col].dtype}"


# ===========================================================================
# Group 4: load_annotation_table
# ===========================================================================


def test_load_annotation_table_tsv_path(tmp_path):
    tx_ids = ["G0000_T0", "G0000_T1", "G0001_T0", "G0001_T1"]
    gene_ids_full = ["G0000", "G0000", "G0001", "G0001"]
    df = _make_annotation_df(gene_ids_full, tx_ids)
    tsv_path = tmp_path / "annot.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    loaded = load_annotation_table(tsv_path)
    assert "feature_id" in loaded.columns
    assert any(c in loaded.columns for c in _ALL_ANNOTATION_COLUMNS)


def test_load_annotation_table_transcript_id_alias(tmp_path):
    df = pd.DataFrame(
        {
            "transcript_id": ["T1", "T2"],
            "gene_id": ["G1", "G1"],
            "first_exon_changed": [True, False],
        }
    )
    loaded = load_annotation_table(df)
    assert "feature_id" in loaded.columns
    assert "transcript_id" not in loaded.columns


def test_load_annotation_table_no_annotation_cols_warns():
    df = pd.DataFrame({"feature_id": ["T1"], "gene_id": ["G1"], "some_other_col": [1.0]})
    with pytest.warns(UserWarning, match="no recognized annotation columns"):
        loaded = load_annotation_table(df)
    assert "feature_id" in loaded.columns


def test_load_annotation_table_missing_feature_id_raises():
    df = pd.DataFrame({"gene_id": ["G1"], "first_exon_changed": [True]})
    with pytest.raises(ValueError, match="feature_id"):
        load_annotation_table(df)


def test_load_annotation_table_isa_alias_cassette_exon():
    df = pd.DataFrame(
        {
            "feature_id": ["T1", "T2"],
            "gene_id": ["G1", "G1"],
            "cassette_exon": ["TRUE", "FALSE"],
        }
    )
    loaded = load_annotation_table(df)
    assert "internal_exon_difference" in loaded.columns
    assert "cassette_exon" not in loaded.columns
    assert isinstance(loaded["internal_exon_difference"].dtype, pd.BooleanDtype)


def test_load_annotation_table_parquet_path(tmp_path):
    df = _make_annotation_df(["G0000", "G0001"], ["G0000_T0", "G0001_T0"])
    parquet_path = tmp_path / "annot.parquet"
    df.to_parquet(parquet_path, index=False)
    loaded = load_annotation_table(parquet_path)
    assert "feature_id" in loaded.columns


# ===========================================================================
# Group 5: annotate_driver_table
# ===========================================================================


def test_annotate_driver_table_appends_columns():
    gene_ids = ["G0000", "G0001", "G0002"]
    tx_ids = ["G0000_T0", "G0000_T1", "G0001_T0", "G0001_T1", "G0002_T0", "G0002_T1"]
    gene_ids_full = ["G0000"] * 2 + ["G0001"] * 2 + ["G0002"] * 2
    annotation = _make_annotation_df(gene_ids_full, tx_ids)
    gene_driver = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "r": [0.9, 0.8, 0.7],
            "pvalue": [0.01, 0.02, 0.03],
            "qvalue": [0.02, 0.03, 0.04],
            "n_samples": [48, 48, 48],
            "missing_fraction": [0.0, 0.0, 0.0],
        }
    )
    result = annotate_driver_table(gene_driver, annotation)
    assert set(gene_driver.columns).issubset(set(result.columns))
    new_cols = set(result.columns) - set(gene_driver.columns)
    assert len(new_cols) > 0


def test_annotate_driver_table_gene_level_any_aggregation():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T_a", "T_b"],
            "gene_id": ["G1", "G1"],
            "first_exon_changed": pd.array([True, False], dtype=pd.BooleanDtype()),
        }
    )
    gene_driver = pd.DataFrame({"gene_id": ["G1"], "r": [0.9]})
    result = annotate_driver_table(gene_driver, annotation)
    assert result.loc[result["gene_id"] == "G1", "first_exon_changed"].iloc[0] == True  # noqa: E712


def test_annotate_driver_table_all_false_stays_false():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T_a", "T_b"],
            "gene_id": ["G2", "G2"],
            "first_exon_changed": pd.array([False, False], dtype=pd.BooleanDtype()),
        }
    )
    gene_driver = pd.DataFrame({"gene_id": ["G2"], "r": [0.7]})
    result = annotate_driver_table(gene_driver, annotation)
    val = result.loc[result["gene_id"] == "G2", "first_exon_changed"].iloc[0]
    assert val is False or val == False  # noqa: E712


def test_annotate_driver_table_unmatched_gene_gets_na():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T1"],
            "gene_id": ["G1"],
            "first_exon_changed": pd.array([True], dtype=pd.BooleanDtype()),
        }
    )
    gene_driver = pd.DataFrame({"gene_id": ["G_UNKNOWN"], "r": [0.5]})
    result = annotate_driver_table(gene_driver, annotation)
    val = result.loc[result["gene_id"] == "G_UNKNOWN", "first_exon_changed"].iloc[0]
    assert pd.isna(val)


def test_annotate_driver_table_no_gene_id_raises():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T1"],
            "first_exon_changed": pd.array([True], dtype=pd.BooleanDtype()),
        }
    )
    gene_driver = pd.DataFrame({"gene_id": ["G1"], "r": [0.9]})
    with pytest.raises(ValueError, match="gene_id"):
        annotate_driver_table(gene_driver, annotation)


def test_annotate_driver_table_no_mutation():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T1"],
            "gene_id": ["G1"],
            "cds_changed": pd.array([True], dtype=pd.BooleanDtype()),
        }
    )
    gene_driver = pd.DataFrame({"gene_id": ["G1"], "r": [0.9]})
    orig_annot_cols = list(annotation.columns)
    orig_driver_cols = list(gene_driver.columns)
    annotate_driver_table(gene_driver, annotation)
    assert list(annotation.columns) == orig_annot_cols
    assert list(gene_driver.columns) == orig_driver_cols


# ===========================================================================
# Group 6: annotate_transcript_table
# ===========================================================================


def test_annotate_transcript_table_appends_columns():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T1", "T2"],
            "gene_id": ["G1", "G1"],
            "first_exon_changed": pd.array([True, False], dtype=pd.BooleanDtype()),
        }
    )
    tx_table = pd.DataFrame(
        {
            "feature_id": ["T1", "T2"],
            "gene_id": ["G1", "G1"],
            "r": [0.9, -0.8],
            "switch_strength": [0.7, 0.6],
        }
    )
    result = annotate_transcript_table(tx_table, annotation)
    assert "first_exon_changed" in result.columns
    assert set(tx_table.columns).issubset(set(result.columns))


def test_annotate_transcript_table_feature_id_join():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T1", "T2"],
            "gene_id": ["G1", "G1"],
            "cds_changed": pd.array([True, False], dtype=pd.BooleanDtype()),
        }
    )
    tx_table = pd.DataFrame(
        {
            "feature_id": ["T1", "T2"],
            "gene_id": ["G1", "G1"],
            "r": [0.9, -0.8],
        }
    )
    result = annotate_transcript_table(tx_table, annotation)
    assert result.loc[result["feature_id"] == "T1", "cds_changed"].iloc[0] == True  # noqa: E712
    assert result.loc[result["feature_id"] == "T2", "cds_changed"].iloc[0] == False  # noqa: E712


def test_annotate_transcript_table_unmatched_feature_gets_na():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T1"],
            "gene_id": ["G1"],
            "first_exon_changed": pd.array([True], dtype=pd.BooleanDtype()),
        }
    )
    tx_table = pd.DataFrame(
        {
            "feature_id": ["T1", "T_UNKNOWN"],
            "gene_id": ["G1", "G1"],
            "r": [0.9, 0.5],
        }
    )
    result = annotate_transcript_table(tx_table, annotation)
    val = result.loc[result["feature_id"] == "T_UNKNOWN", "first_exon_changed"].iloc[0]
    assert pd.isna(val)


def test_annotate_transcript_table_empty_returns_unchanged():
    annotation = pd.DataFrame(
        {
            "feature_id": ["T1"],
            "gene_id": ["G1"],
            "first_exon_changed": pd.array([True], dtype=pd.BooleanDtype()),
        }
    )
    empty_table = pd.DataFrame(columns=["feature_id", "gene_id", "r"])
    result = annotate_transcript_table(empty_table, annotation)
    assert result.empty
    assert list(result.columns) == list(empty_table.columns)


# ===========================================================================
# Group 7: explain_module integration
# ===========================================================================


from isograph.explain import ExplainResult, explain_module


def test_explain_module_with_annotation_table(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)

    tx_ids = list(feature_table.columns)
    gene_ids = [fid.rsplit("_T", 1)[0] for fid in tx_ids]
    annotation = _make_annotation_df(gene_ids, tx_ids)

    results = explain_module(
        artifact_dir,
        feature_table,
        feature_meta,
        module_ids=["M000"],
        annotation_table=annotation,
    )
    result = results["M000"]
    assert isinstance(result, ExplainResult)
    # gene_driver_table should have at least one new annotation column
    new_cols = set(result.gene_driver_table.columns) - {
        "gene_id",
        "r",
        "pvalue",
        "qvalue",
        "n_samples",
        "missing_fraction",
    }
    assert len(new_cols) > 0
    # transcript_polarity_table should also have annotation columns (if non-empty)
    if not result.transcript_polarity_table.empty:
        new_tx_cols = set(result.transcript_polarity_table.columns) - {
            "feature_id",
            "gene_id",
            "r",
            "switch_strength",
            "pvalue",
            "qvalue",
            "n_samples",
            "missing_fraction",
        }
        assert len(new_tx_cols) > 0


def test_explain_module_annotation_manifest_fields(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    output_dir = tmp_path / "out"

    tx_ids = list(feature_table.columns)
    gene_ids = [fid.rsplit("_T", 1)[0] for fid in tx_ids]
    annotation = _make_annotation_df(gene_ids, tx_ids)

    explain_module(
        artifact_dir,
        feature_table,
        feature_meta,
        output_dir=output_dir,
        annotation_table=annotation,
    )
    manifest = json.loads((output_dir / "module_explanation_manifest.json").read_text())
    assert manifest["annotation_provided"] is True
    assert isinstance(manifest["annotation_columns"], list)
    assert len(manifest["annotation_columns"]) > 0
    for col in manifest["annotation_columns"]:
        assert col in _ALL_ANNOTATION_COLUMNS


def test_explain_module_no_annotation_regression(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    output_dir = tmp_path / "out"

    results = explain_module(
        artifact_dir,
        feature_table,
        feature_meta,
        output_dir=output_dir,
    )
    manifest = json.loads((output_dir / "module_explanation_manifest.json").read_text())
    assert manifest["annotation_provided"] is False
    assert manifest["annotation_columns"] == []

    # Core tables should have no annotation columns
    result = results["M000"]
    for col in _ALL_ANNOTATION_COLUMNS:
        assert col not in result.gene_driver_table.columns
