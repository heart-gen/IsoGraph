"""Dataset artifact I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from isograph.utils import ensure_dir, write_json
from isograph.validation import DatasetManifest, FeatureTableSpec, LoadedDataset, MatrixSpec


@dataclass
class DatasetBundle:
    manifest: DatasetManifest
    sample_table: pd.DataFrame
    feature_tables: dict[str, pd.DataFrame]
    matrices: dict[str, np.ndarray]
    truth_tables: dict[str, pd.DataFrame]


def save_dense_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savez_compressed(path, data=matrix)


def load_dense_matrix(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        return loaded["data"]


def save_dataset_bundle(bundle: DatasetBundle, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    bundle.sample_table.to_parquet(output_dir / bundle.manifest.sample_table, index=False)

    for spec in bundle.manifest.feature_tables:
        table = bundle.feature_tables[spec.kind]
        table.to_parquet(output_dir / spec.filename, index=False)

    for spec in bundle.manifest.matrices:
        save_dense_matrix(output_dir / spec.filename, bundle.matrices[spec.assay_name])

    for truth_name, table in bundle.truth_tables.items():
        table.to_parquet(output_dir / truth_name, index=False)

    write_json(output_dir / "manifest.json", bundle.manifest.model_dump())
    return output_dir


def load_dataset_bundle(dataset_dir: Path) -> DatasetBundle:
    manifest = DatasetManifest.model_validate_json((dataset_dir / "manifest.json").read_text())
    sample_table = pd.read_parquet(dataset_dir / manifest.sample_table)
    feature_tables = {
        spec.kind: pd.read_parquet(dataset_dir / spec.filename) for spec in manifest.feature_tables
    }
    matrices = {
        spec.assay_name: load_dense_matrix(dataset_dir / spec.filename) for spec in manifest.matrices
    }
    truth_tables = {name: pd.read_parquet(dataset_dir / name) for name in manifest.truth_tables}
    return DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables=feature_tables,
        matrices=matrices,
        truth_tables=truth_tables,
    )


def describe_dataset(dataset_dir: Path) -> LoadedDataset:
    bundle = load_dataset_bundle(dataset_dir)
    gene_table = bundle.feature_tables.get("gene")
    return LoadedDataset(
        manifest_path=dataset_dir / "manifest.json",
        n_samples=len(bundle.sample_table),
        n_genes=0 if gene_table is None else len(gene_table),
        available_assays=sorted(bundle.matrices),
    )


def build_feature_spec(kind: str, filename: str, table: pd.DataFrame) -> FeatureTableSpec:
    return FeatureTableSpec(kind=kind, filename=filename, n_rows=len(table))


def build_matrix_spec(name: str, filename: str, matrix: np.ndarray) -> MatrixSpec:
    return MatrixSpec(
        assay_name=name,
        filename=filename,
        n_features=int(matrix.shape[0]),
        n_samples=int(matrix.shape[1]),
    )
