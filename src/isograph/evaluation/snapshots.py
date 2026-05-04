"""Snapshot save and deterministic comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from isograph.features.channels import feature_sample_columns
from isograph.models.base import FitArtifacts
from isograph.utils import dataclass_to_jsonable, ensure_dir, write_json
from isograph.workflow.config import BaselineModelConfig

_ISOGRAPH_VERSION = "0.1.0"

# Files that must be present in every snapshot directory.
SNAPSHOT_FILES = {
    "manifest.json",
    "metrics.json",
    "module_summary.tsv",
    "switch_features.parquet",
    "run_config.yaml",
}


def _module_summary(module_table: pd.DataFrame) -> pd.DataFrame:
    if module_table.empty:
        return pd.DataFrame(columns=["module_id", "n_genes"])
    summary = (
        module_table.groupby("module_id", sort=True)["gene_id"]
        .count()
        .reset_index()
        .rename(columns={"gene_id": "n_genes"})
    )
    return summary.sort_values("module_id").reset_index(drop=True)


def _sorted_feature_scores(feature_scores: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [column for column in ("gene_id", "feature_type", "feature_id") if column in feature_scores.columns]
    return feature_scores.sort_values(sort_cols).reset_index(drop=True)


def save_snapshot(
    fit_artifacts: FitArtifacts,
    model_config: BaselineModelConfig,
    metrics: dict[str, Any],
    output_dir: Path,
    snapshot_name: str = "",
    dataset_name: str = "",
) -> Path:
    """Save a model-run snapshot to *output_dir*.

    Required keys in *metrics*: ``n_modules``, ``n_edges``.
    Optional keys: ``recovery`` (float | None), ``runtime_seconds`` (float).

    Returns the resolved output directory path.
    """
    ensure_dir(output_dir)

    # manifest.json
    manifest: dict[str, Any] = {
        "snapshot_name": snapshot_name,
        "dataset_name": dataset_name,
        "isograph_version": _ISOGRAPH_VERSION,
        "model": model_config.name,
    }
    write_json(output_dir / "manifest.json", manifest)

    # metrics.json
    metrics_payload: dict[str, Any] = {
        "n_modules": int(metrics.get("n_modules", 0)),
        "n_edges": int(metrics.get("n_edges", 0)),
        "recovery": metrics.get("recovery"),
        "runtime_seconds": metrics.get("runtime_seconds"),
    }
    write_json(output_dir / "metrics.json", metrics_payload)

    # module_summary.tsv
    summary = _module_summary(fit_artifacts.module_table)
    summary.to_csv(output_dir / "module_summary.tsv", sep="\t", index=False)

    # switch_features.parquet
    sorted_scores = _sorted_feature_scores(fit_artifacts.feature_scores)
    sorted_scores.to_parquet(output_dir / "switch_features.parquet", index=False)

    # eigengene_table.parquet (optional — present when modules exist)
    if fit_artifacts.eigengene_table is not None:
        fit_artifacts.eigengene_table.to_parquet(output_dir / "eigengene_table.parquet", index=False)
    if fit_artifacts.module_gene_roles is not None:
        fit_artifacts.module_gene_roles.to_parquet(output_dir / "module_gene_roles.parquet", index=False)

    # run_config.yaml
    config_dict = dataclass_to_jsonable(model_config)
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(config_dict, fh, default_flow_style=False, sort_keys=True)

    return output_dir


def compare_snapshot_dirs(
    reference_dir: Path,
    candidate_dir: Path,
    recovery_tolerance: float = 1e-6,
    feature_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare two snapshot directories and return a diff report.

    Returns a dict with keys:
      - ``passed`` (bool)
      - ``differences`` (list[str]) — empty when passed
      - ``reference`` / ``candidate`` (str) — resolved paths
    """
    differences: list[str] = []

    # Verify both directories exist and are complete
    for label, directory in (("reference", reference_dir), ("candidate", candidate_dir)):
        if not directory.is_dir():
            differences.append(f"{label} directory not found: {directory}")
        else:
            missing = SNAPSHOT_FILES - {p.name for p in directory.iterdir()}
            for name in sorted(missing):
                differences.append(f"{label} snapshot is missing file: {name}")

    if differences:
        return {"passed": False, "differences": differences,
                "reference": str(reference_dir), "candidate": str(candidate_dir)}

    # Compare metrics.json
    ref_metrics: dict[str, Any] = json.loads((reference_dir / "metrics.json").read_text())
    cand_metrics: dict[str, Any] = json.loads((candidate_dir / "metrics.json").read_text())

    for key in ("n_modules", "n_edges"):
        ref_val = ref_metrics.get(key)
        cand_val = cand_metrics.get(key)
        if ref_val != cand_val:
            differences.append(f"metrics.{key} differs: reference={ref_val} candidate={cand_val}")

    ref_rec = ref_metrics.get("recovery")
    cand_rec = cand_metrics.get("recovery")
    if (ref_rec is None) != (cand_rec is None):
        differences.append(f"metrics.recovery nullability differs: reference={ref_rec} candidate={cand_rec}")
    elif ref_rec is not None and cand_rec is not None:
        if abs(float(ref_rec) - float(cand_rec)) > recovery_tolerance:
            differences.append(
                f"metrics.recovery differs beyond tolerance={recovery_tolerance}: "
                f"reference={ref_rec:.8f} candidate={cand_rec:.8f}"
            )

    # Compare module_summary.tsv
    ref_summary = pd.read_csv(reference_dir / "module_summary.tsv", sep="\t")
    cand_summary = pd.read_csv(candidate_dir / "module_summary.tsv", sep="\t")
    if list(ref_summary.columns) != list(cand_summary.columns):
        differences.append("module_summary.tsv column names differ")
    elif ref_summary.shape != cand_summary.shape:
        differences.append(
            f"module_summary.tsv shape differs: reference={ref_summary.shape} candidate={cand_summary.shape}"
        )
    else:
        ref_sorted = ref_summary.sort_values("module_id").reset_index(drop=True)
        cand_sorted = cand_summary.sort_values("module_id").reset_index(drop=True)
        if not ref_sorted.equals(cand_sorted):
            differences.append("module_summary.tsv contents differ")

    # Compare switch_features.parquet
    ref_feat = pd.read_parquet(reference_dir / "switch_features.parquet")
    cand_feat = pd.read_parquet(candidate_dir / "switch_features.parquet")
    if list(ref_feat.columns) != list(cand_feat.columns):
        differences.append("switch_features.parquet column names differ")
    elif ref_feat.shape != cand_feat.shape:
        differences.append(
            f"switch_features.parquet shape differs: reference={ref_feat.shape} candidate={cand_feat.shape}"
        )
    else:
        ref_feat_s = _sorted_feature_scores(ref_feat)
        cand_feat_s = _sorted_feature_scores(cand_feat)
        numeric_cols = feature_sample_columns(ref_feat_s)
        try:
            np.testing.assert_allclose(
                ref_feat_s[numeric_cols].to_numpy(dtype=float),
                cand_feat_s[numeric_cols].to_numpy(dtype=float),
                atol=feature_tolerance,
                rtol=0.0,
            )
        except AssertionError as exc:
            differences.append(f"switch_features.parquet numeric values differ: {exc}")
        id_cols = [column for column in ("feature_id", "gene_id", "feature_type") if column in ref_feat_s.columns]
        for column in id_cols:
            if not ref_feat_s[column].equals(cand_feat_s[column]):
                differences.append(f"switch_features.parquet {column} ordering differs")

    return {
        "passed": len(differences) == 0,
        "differences": differences,
        "reference": str(reference_dir),
        "candidate": str(candidate_dir),
    }
