"""Synthetic benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from isograph.io.artifacts import (
    DatasetBundle,
    build_feature_spec,
    build_matrix_spec,
    save_dataset_bundle,
)
from isograph.validation import DatasetManifest


@dataclass
class SyntheticDatasetSpec:
    name: str
    n_genes: int
    n_samples: int
    n_modules: int
    seed: int


def _module_assignments(n_genes: int, n_modules: int) -> np.ndarray:
    modules = np.repeat(np.arange(n_modules), np.ceil(n_genes / n_modules))[:n_genes]
    return modules.astype(int)


def _generate_dataset(spec: SyntheticDatasetSpec, suite_name: str, description: str) -> DatasetBundle:
    rng = np.random.default_rng(spec.seed)
    sample_ids = [f"S{i:03d}" for i in range(spec.n_samples)]
    gene_ids = [f"G{i:04d}" for i in range(spec.n_genes)]
    module_ids = _module_assignments(spec.n_genes, spec.n_modules)

    dx = np.array(["Control"] * (spec.n_samples // 2) + ["SCZD"] * (spec.n_samples - spec.n_samples // 2))
    age = np.linspace(30, 70, spec.n_samples) + rng.normal(0, 2, spec.n_samples)
    sex = np.where(np.arange(spec.n_samples) % 2 == 0, "F", "M")
    sample_table = pd.DataFrame({"sample_id": sample_ids, "Dx": dx, "Age": age, "Sex": sex})

    module_latent = rng.normal(size=(spec.n_modules, spec.n_samples))
    module_latent += (dx == "SCZD")[None, :] * np.linspace(0.8, 1.2, spec.n_modules)[:, None]
    module_latent += ((age - age.mean()) / age.std())[None, :] * np.linspace(0.4, 0.7, spec.n_modules)[:, None]

    gene_totals = rng.gamma(shape=10.0, scale=30.0, size=(spec.n_genes, spec.n_samples))
    switch_signal = np.vstack([module_latent[module_ids[index]] for index in range(spec.n_genes)])
    p1 = 1.0 / (1.0 + np.exp(-switch_signal))
    p2 = 1.0 - p1
    transcript_counts = np.zeros((spec.n_genes * 2, spec.n_samples), dtype=float)
    transcript_feature_rows: list[dict[str, object]] = []
    for index, gene_id in enumerate(gene_ids):
        total = np.maximum(gene_totals[index], 1.0)
        tx1 = np.floor(total * p1[index])
        tx2 = np.maximum(total - tx1, 0.0)
        transcript_counts[index * 2] = tx1
        transcript_counts[index * 2 + 1] = tx2
        transcript_feature_rows.extend(
            [
                {"transcript_id": f"{gene_id}_T1", "gene_id": gene_id, "length": 1000},
                {"transcript_id": f"{gene_id}_T2", "gene_id": gene_id, "length": 950},
            ]
        )

    gene_counts = transcript_counts.reshape(spec.n_genes, 2, spec.n_samples).sum(axis=1)
    psi = np.clip(p1 + rng.normal(0, 0.03, size=p1.shape), 1e-4, 1 - 1e-4)
    gene_table = pd.DataFrame(
        {"gene_id": gene_ids, "chrom": "chrSynthetic", "start": np.arange(spec.n_genes), "end": np.arange(spec.n_genes) + 100}
    )
    transcript_table = pd.DataFrame(transcript_feature_rows)
    psi_table = pd.DataFrame({"psi_uid": [f"PSI_{gene_id}" for gene_id in gene_ids], "gene_id": gene_ids})
    truth_modules = pd.DataFrame({"gene_id": gene_ids, "module_id": module_ids})
    truth_switch = pd.DataFrame({"gene_id": gene_ids, "has_switch": True})

    manifest = DatasetManifest(
        dataset_name=spec.name,
        suite_name=suite_name,
        description=description,
        sample_table="samples.parquet",
        feature_tables=[
            build_feature_spec("gene", "genes.parquet", gene_table),
            build_feature_spec("transcript", "transcripts.parquet", transcript_table),
            build_feature_spec("psi", "psi.parquet", psi_table),
            build_feature_spec("truth_module", "truth_modules.parquet", truth_modules),
            build_feature_spec("truth_switch", "truth_switch.parquet", truth_switch),
        ],
        matrices=[
            build_matrix_spec("gene_counts", "gene_counts.npz", gene_counts),
            build_matrix_spec("transcript_counts", "transcript_counts.npz", transcript_counts),
            build_matrix_spec("psi", "psi.npz", psi),
        ],
        provenance={"generator": "synthetic_v1", "seed": str(spec.seed)},
        truth_tables=["truth_modules.parquet", "truth_switch.parquet"],
    )
    return DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables={
            "gene": gene_table,
            "transcript": transcript_table,
            "psi": psi_table,
            "truth_module": truth_modules,
            "truth_switch": truth_switch,
        },
        matrices={"gene_counts": gene_counts, "transcript_counts": transcript_counts, "psi": psi},
        truth_tables={"truth_modules.parquet": truth_modules, "truth_switch.parquet": truth_switch},
    )


def generate_core_suite(root: Path, seed: int) -> list[Path]:
    suite_dir = root / "core_v1"
    toy = _generate_dataset(
        SyntheticDatasetSpec(name="toy_v1", n_genes=24, n_samples=48, n_modules=2, seed=seed),
        suite_name="core_v1",
        description="Toy deterministic simulation with exact switch truth.",
    )
    medium = _generate_dataset(
        SyntheticDatasetSpec(
            name="medium_v1", n_genes=400, n_samples=240, n_modules=8, seed=seed + 1
        ),
        suite_name="core_v1",
        description="Medium deterministic simulation with planted switch modules.",
    )
    return [
        save_dataset_bundle(toy, suite_dir / "toy_v1"),
        save_dataset_bundle(medium, suite_dir / "medium_v1"),
    ]
