"""Run Stage 8 explain checks on multiplex stress benchmark artifacts.

This script adapts benchmark snapshots into explain-module inputs without
modifying the benchmark outputs. It reports switch-detection metrics against the
multiplex truth tables for each backend/fixture.

Usage:
    python scripts/stress_multiplex_explain.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isograph.explain import explain_module
from isograph.io.artifacts import load_dataset_bundle

DEFAULT_STAGES = {
    "vae": "stress_multiplex_vae",
    "graph": "stress_multiplex_graph",
    "latent": "stress_multiplex_latent",
    "wgcna": "stress_multiplex_wgcna",
}


def _build_transcript_usage(
    transcript_counts: np.ndarray,
    transcript_table: pd.DataFrame,
    sample_ids: list[str],
) -> pd.DataFrame:
    usage = np.zeros((len(transcript_table), len(sample_ids)), dtype=float)
    for _, group in transcript_table.groupby("gene_id", sort=False):
        indices = group.index.to_numpy()
        block = transcript_counts[indices]
        totals = block.sum(axis=0, keepdims=True)
        usage[indices] = block / np.maximum(totals, 1.0)
    return pd.DataFrame(
        usage.T,
        index=sample_ids,
        columns=transcript_table["transcript_id"].astype(str).tolist(),
    )


def _feature_meta(transcript_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_id": transcript_table["transcript_id"].astype(str).tolist(),
            "gene_id": transcript_table["gene_id"].astype(str).tolist(),
            "transcript_id": transcript_table["transcript_id"].astype(str).tolist(),
            "feature_type": "transcript_usage",
        }
    )


def _auc(scores: list[float], labels: list[int]) -> float | None:
    arr_scores = np.asarray(scores, dtype=float)
    arr_labels = np.asarray(labels, dtype=int)
    pos = arr_scores[arr_labels == 1]
    neg = arr_scores[arr_labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    total = 0.0
    for value in pos:
        total += float((value > neg).sum()) + 0.5 * float((value == neg).sum())
    return total / (len(pos) * len(neg))


def _metrics(results: dict[str, Any], truth_switch: pd.DataFrame) -> dict[str, Any]:
    switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"].astype(str))
    abs_r_switch: list[float] = []
    abs_r_nonswitch: list[float] = []
    switch_strength_scores: list[float] = []
    switch_strength_labels: list[int] = []
    delta_switch: list[float] = []
    delta_nonswitch: list[float] = []
    two_isoform_total = 0
    two_isoform_high_r = 0

    for result in results.values():
        drivers = result.gene_driver_table
        finite = drivers["r"].notna()
        for _, row in drivers.loc[finite].iterrows():
            gene_id = str(row["gene_id"])
            target = abs_r_switch if gene_id in switch_set else abs_r_nonswitch
            target.append(abs(float(row["r"])))

        polarity = result.transcript_polarity_table
        if not polarity.empty:
            for gene_id, group in polarity.groupby("gene_id"):
                gene = str(gene_id)
                strength = float(group["switch_strength"].iloc[0])
                switch_strength_scores.append(strength)
                switch_strength_labels.append(1 if gene in switch_set else 0)
                if gene in switch_set and len(group) == 2:
                    max_abs_r = float(group["r"].abs().max())
                    if math.isfinite(max_abs_r):
                        two_isoform_total += 1
                        if max_abs_r >= 0.5:
                            two_isoform_high_r += 1

        high_low = result.high_vs_low_table
        finite_delta = high_low["delta"].notna()
        for _, row in high_low.loc[finite_delta].iterrows():
            gene_id = str(row["gene_id"])
            target = delta_switch if gene_id in switch_set else delta_nonswitch
            target.append(abs(float(row["delta"])))

    switch_strength_auc = _auc(switch_strength_scores, switch_strength_labels)
    return {
        "n_modules_explained": len(results),
        "n_driver_genes": sum(len(r.gene_driver_table) for r in results.values()),
        "mean_abs_r_switching": float(np.mean(abs_r_switch)) if abs_r_switch else None,
        "mean_abs_r_nonswitching": float(np.mean(abs_r_nonswitch)) if abs_r_nonswitch else None,
        "switch_strength_auc": switch_strength_auc,
        "two_isoform_high_r_fraction": (
            two_isoform_high_r / two_isoform_total if two_isoform_total else None
        ),
        "mean_abs_delta_switching": float(np.mean(delta_switch)) if delta_switch else None,
        "mean_abs_delta_nonswitching": float(np.mean(delta_nonswitch)) if delta_nonswitch else None,
    }


def _prepare_artifact_dir(source_dir: Path, tmp_dir: Path) -> None:
    roles = source_dir / "module_gene_roles.parquet"
    scores = source_dir / "switch_features.parquet"
    if not roles.exists():
        raise FileNotFoundError(f"Missing module_gene_roles.parquet: {roles}")
    if not scores.exists():
        raise FileNotFoundError(f"Missing switch_features.parquet: {scores}")

    modules = pd.read_parquet(roles)[["gene_id", "module_id"]].drop_duplicates()
    feature_scores = pd.read_parquet(scores)

    modules.to_parquet(tmp_dir / "modules.parquet", index=False)
    feature_scores.to_parquet(tmp_dir / "feature_scores.parquet", index=False)


def run(
    dataset_root: Path,
    artifacts_root: Path,
    output_dir: Path,
    stages: dict[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for backend, stage in stages.items():
        stage_dir = artifacts_root / "benchmarks" / stage
        if not stage_dir.exists():
            raise FileNotFoundError(f"Missing benchmark artifact directory: {stage_dir}")
        for fixture_dir in sorted(stage_dir.iterdir()):
            if not fixture_dir.is_dir():
                continue
            dataset_dir = dataset_root / "multiplex_v1" / fixture_dir.name
            bundle = load_dataset_bundle(dataset_dir)
            sample_ids = bundle.sample_table["sample_id"].astype(str).tolist()
            transcript_table = bundle.feature_tables["transcript"].copy()
            transcript_table["transcript_id"] = transcript_table["transcript_id"].astype(str)
            transcript_table["gene_id"] = transcript_table["gene_id"].astype(str)

            feature_table = _build_transcript_usage(
                bundle.matrices["transcript_counts"], transcript_table, sample_ids
            )
            feature_meta = _feature_meta(transcript_table)
            truth_switch = bundle.truth_tables["truth_switch.parquet"].copy()
            truth_switch["gene_id"] = truth_switch["gene_id"].astype(str)

            with tempfile.TemporaryDirectory() as tmp_name:
                tmp_dir = Path(tmp_name)
                _prepare_artifact_dir(fixture_dir, tmp_dir)
                out = output_dir / backend / fixture_dir.name
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    results = explain_module(
                        artifact_dir=tmp_dir,
                        feature_table=feature_table,
                        feature_meta=feature_meta,
                        output_dir=out,
                    )
            row = {
                "backend": backend,
                "stage": stage,
                "dataset": fixture_dir.name,
                **_metrics(results, truth_switch),
            }
            rows.append(row)

    report = {"suite": "multiplex_v1", "results": rows}
    (output_dir / "stress_multiplex_explain_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="benchmarks/datasets")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--output-dir", default="artifacts/explain/stress_multiplex")
    args = parser.parse_args(argv)
    report = run(
        dataset_root=Path(args.dataset_root),
        artifacts_root=Path(args.artifacts_root),
        output_dir=Path(args.output_dir),
        stages=DEFAULT_STAGES,
    )
    print(f"{len(report['results'])} explain rows")
    print(Path(args.output_dir) / "stress_multiplex_explain_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
