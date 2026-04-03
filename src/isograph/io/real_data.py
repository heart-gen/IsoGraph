"""Pure-Python real-data freeze pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from isograph.io.artifacts import (
    DatasetBundle,
    build_feature_spec,
    build_matrix_spec,
    save_dataset_bundle,
)
from isograph.utils import ensure_dir
from isograph.validation import DatasetManifest, FreezeSelectionConfig
from isograph.workflow.config import RealDataFreezeConfig


def _read_table(path: Path, sep: str = "\t") -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=sep, compression="infer")
    except (OSError, ValueError):
        return pd.read_csv(path, sep=sep, compression=None)


def _select_samples(config: RealDataFreezeConfig) -> pd.DataFrame:
    sample_df = _read_table(config.phenotype_tsv)
    keep = (
        sample_df["Region"].eq("Caudate")
        & sample_df["Race"].eq("AA")
        & sample_df["Age"].gt(17)
        & sample_df["dropped"].eq("f")
        & sample_df["Dx"].isin(config.allowed_diagnoses)
    )
    sample_df = sample_df.loc[keep].copy()
    sample_df = sample_df.sort_values("RNum").reset_index(drop=True)
    sample_df["Age"] = sample_df["Age"].astype(float)
    return sample_df


def _join_ancestry(sample_df: pd.DataFrame, ancestry_path: Path) -> pd.DataFrame:
    if not ancestry_path.exists():
        return sample_df
    ancestry = pd.read_csv(ancestry_path, sep=r"\s+", engine="python")
    if "#FID" in ancestry.columns:
        ancestry = ancestry.rename(columns={"#FID": "BrNum"})
    sample_df = sample_df.merge(ancestry, on="BrNum", how="left")
    return sample_df


def _load_count_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def _sample_columns(sample_df: pd.DataFrame, columns: list[str]) -> list[str]:
    rnums = set(sample_df["RNum"])
    return [column for column in columns if column in rnums]


def _standardize_annotation_columns(table: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.lower() for column in table.columns}
    return table.rename(columns=renamed)


def _choose_gene_panel(gene_counts: pd.DataFrame, sample_cols: list[str], panel_size: int) -> pd.Index:
    totals = gene_counts[sample_cols].to_numpy(dtype=float)
    score = totals.mean(axis=1) * (1.0 + totals.var(axis=1))
    top_idx = np.argsort(score)[::-1][:panel_size]
    return gene_counts.iloc[top_idx]["Geneid"]


def _subset_matrix(
    table: pd.DataFrame,
    feature_id_col: str,
    sample_cols: list[str],
    selected_ids: set[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    subset = table.loc[table[feature_id_col].isin(selected_ids)].copy()
    subset = subset.reset_index(drop=True)
    matrix = subset[sample_cols].to_numpy(dtype=float).T.T
    return subset, matrix


def freeze_real_dataset(config: RealDataFreezeConfig, suite_dir: Path) -> Path:
    FreezeSelectionConfig(
        gene_panel_size=config.gene_panel_size, allowed_diagnoses=config.allowed_diagnoses
    )
    suite_dir = ensure_dir(suite_dir)
    output_dir = suite_dir / config.output_name
    counts_root = Path(config.counts_root)
    annot_root = Path(config.annotations_root)

    samples = _join_ancestry(_select_samples(config), Path(config.ancestry_tsv))
    sample_cols = list(samples["RNum"])

    gene_counts = _load_count_table(counts_root / "gene-counts.tsv")
    sample_cols = _sample_columns(samples, list(gene_counts.columns))
    samples = samples.loc[samples["RNum"].isin(sample_cols)].copy().reset_index(drop=True)

    selected_genes = set(_choose_gene_panel(gene_counts, sample_cols, config.gene_panel_size))
    gene_subset = gene_counts.loc[gene_counts["Geneid"].isin(selected_genes)].copy()
    gene_matrix = gene_subset[sample_cols].to_numpy(dtype=float)

    tx_counts = _load_count_table(counts_root / "tx-counts.tsv")
    tx_annot = _standardize_annotation_columns(_read_table(annot_root / "transcript_annotation.tsv.gz"))
    tx_feature_col = "Name" if "Name" in tx_counts.columns else "transcript_id"
    tx_gene_map = tx_annot.loc[tx_annot["gene_id"].isin(selected_genes), ["transcript_id", "gene_id"]]
    selected_tx = set(tx_gene_map["transcript_id"])
    tx_subset = tx_counts.loc[tx_counts[tx_feature_col].isin(selected_tx)].copy()
    tx_matrix = tx_subset[sample_cols].to_numpy(dtype=float)
    tx_subset = tx_subset.rename(columns={tx_feature_col: "transcript_id"})
    tx_subset = tx_subset.merge(tx_gene_map, on="transcript_id", how="left")

    psi_counts = _load_count_table(counts_root / "psi-events.tsv.gz")
    psi_annot = _standardize_annotation_columns(_read_table(annot_root / "psi_annotation.tsv.gz"))
    psi_annot = psi_annot.loc[psi_annot["gene_id"].isin(selected_genes)].copy()
    psi_key_cols = ["gene_id", "event_info", "event_type", "chr", "strand"]
    available_key_cols = [col for col in psi_key_cols if col in psi_counts.columns and col in psi_annot.columns]
    psi_subset = psi_counts.merge(psi_annot, on=available_key_cols, how="inner")
    psi_matrix = psi_subset[sample_cols].to_numpy(dtype=float)
    if "psi_uid" not in psi_subset.columns:
        psi_subset["psi_uid"] = [f"psi_{index}" for index in range(len(psi_subset))]

    jxn_counts = _load_count_table(counts_root / "junction_matrix.parquet")
    jxn_annot = _standardize_annotation_columns(_read_table(annot_root / "jxn_annotation.tsv.gz"))
    gene_name_map = (
        _standardize_annotation_columns(_read_table(annot_root / "gene_annotation.tsv.gz"))[
            ["gene_id", "gene_name"]
        ]
        .drop_duplicates()
        .rename(columns={"gene_name": "gene_names"})
    )
    jxn_annot = jxn_annot.merge(gene_name_map, on="gene_names", how="left")
    jxn_annot = jxn_annot.loc[jxn_annot["gene_id"].isin(selected_genes)].copy()
    if "junction_name" in jxn_counts.columns:
        jxn_subset = jxn_counts.merge(
            jxn_annot[["junction_name", "gene_id"]].drop_duplicates(),
            on="junction_name",
            how="inner",
        )
    else:
        jxn_subset = jxn_counts.iloc[0:0].copy()
        jxn_subset["gene_id"] = []
    jxn_matrix = jxn_subset[sample_cols].to_numpy(dtype=float) if len(jxn_subset) else np.zeros((0, len(sample_cols)))

    gene_feature_table = gene_subset[["Geneid", "Chr", "Start", "End", "Strand", "Length"]].rename(
        columns={"Geneid": "gene_id", "Chr": "chrom", "Start": "start", "End": "end", "Strand": "strand", "Length": "length"}
    )
    tx_feature_table = tx_subset[["transcript_id", "gene_id", "Length", "EffectiveLength"]].rename(
        columns={"Length": "length", "EffectiveLength": "effective_length"}
    )
    psi_feature_cols = [col for col in ["psi_uid", "gene_id", "event_type", "event_info", "chr", "strand"] if col in psi_subset.columns]
    psi_feature_table = psi_subset[psi_feature_cols].drop_duplicates().reset_index(drop=True)
    if len(jxn_subset):
        jxn_feature_cols = [col for col in ["junction_name", "gene_id"] if col in jxn_subset.columns]
        jxn_feature_table = jxn_subset[jxn_feature_cols].drop_duplicates().reset_index(drop=True)
    else:
        jxn_feature_table = pd.DataFrame(columns=["junction_name", "gene_id"])

    manifest = DatasetManifest(
        dataset_name=config.output_name,
        suite_name=suite_dir.name,
        description="Frozen adult AA caudate subset with Control and SCZD samples only.",
        sample_table="samples.parquet",
        feature_tables=[
            build_feature_spec("gene", "genes.parquet", gene_feature_table),
            build_feature_spec("transcript", "transcripts.parquet", tx_feature_table),
            build_feature_spec("psi", "psi.parquet", psi_feature_table),
            build_feature_spec("junction", "junctions.parquet", jxn_feature_table),
        ],
        matrices=[
            build_matrix_spec("gene_counts", "gene_counts.npz", gene_matrix),
            build_matrix_spec("transcript_counts", "transcript_counts.npz", tx_matrix),
            build_matrix_spec("psi", "psi.npz", psi_matrix),
            build_matrix_spec("junction_counts", "junction_counts.npz", jxn_matrix),
        ],
        provenance={
            "counts_root": str(counts_root),
            "annotations_root": str(annot_root),
            "phenotype_tsv": str(config.phenotype_tsv),
            "selection": "Region=Caudate,Race=AA,Age>17,dropped=f,Dx in {Control,SCZD}",
        },
        truth_tables=[],
    )
    bundle = DatasetBundle(
        manifest=manifest,
        sample_table=samples,
        feature_tables={
            "gene": gene_feature_table,
            "transcript": tx_feature_table,
            "psi": psi_feature_table,
            "junction": jxn_feature_table,
        },
        matrices={
            "gene_counts": gene_matrix,
            "transcript_counts": tx_matrix,
            "psi": psi_matrix,
            "junction_counts": jxn_matrix,
        },
        truth_tables={},
    )
    return save_dataset_bundle(bundle, output_dir)
