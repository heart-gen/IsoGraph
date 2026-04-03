"""Benchmark runner."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json

import pandas as pd

from isograph.benchmarks.synthetic import generate_core_suite
from isograph.evaluation.metrics import module_recovery_score
from isograph.evaluation.tracking import tracking_run
from isograph.io.artifacts import describe_dataset, load_dataset_bundle
from isograph.io.real_data import freeze_real_dataset
from isograph.models.baseline import BaselineNetworkModel
from isograph.utils import ensure_dir, write_json
from isograph.workflow.config import BenchmarkCommandConfig


def prepare_core_suite(config: BenchmarkCommandConfig) -> list[Path]:
    dataset_root = ensure_dir(config.dataset_root)
    paths = generate_core_suite(dataset_root, config.seed)
    paths.append(freeze_real_dataset(config.real_data, dataset_root / config.dataset_suite))
    return paths


def benchmark(config: BenchmarkCommandConfig) -> Path:
    paths = prepare_core_suite(config)
    report_rows = []
    model = BaselineNetworkModel(config.model)
    with tracking_run(config.tracking_uri, run_name=f"{config.dataset_suite}-{config.model.name}"):
        for dataset_dir in paths:
            start = perf_counter()
            bundle = load_dataset_bundle(dataset_dir)
            artifacts = model.fit(
                transcript_counts=bundle.matrices["transcript_counts"],
                transcript_table=bundle.feature_tables["transcript"],
                sample_table=bundle.sample_table,
            )
            elapsed = perf_counter() - start
            truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
            recovery = module_recovery_score(artifacts.module_table, truth) if not truth.empty else None
            report_rows.append(
                {
                    "dataset": bundle.manifest.dataset_name,
                    "n_samples": len(bundle.sample_table),
                    "n_genes": len(bundle.feature_tables.get("gene", [])),
                    "n_modules": 0 if artifacts.module_table.empty else artifacts.module_table["module_id"].nunique(),
                    "n_edges": len(artifacts.edge_table),
                    "recovery": recovery,
                    "runtime_seconds": elapsed,
                }
            )
    report = {"results": report_rows, "suite": config.dataset_suite, "model": config.model.name}
    report_dir = ensure_dir(config.report_root)
    output_path = report_dir / f"{config.dataset_suite}-benchmark.json"
    write_json(output_path, report)
    return output_path


def compare_reports(left_report: Path, right_report: Path, output_path: Path) -> Path:
    left = pd.DataFrame(json.loads(left_report.read_text())["results"])
    right = pd.DataFrame(json.loads(right_report.read_text())["results"])
    merged = left.merge(right, on="dataset", suffixes=("_left", "_right"))
    merged["delta_recovery"] = merged["recovery_right"] - merged["recovery_left"]
    merged["delta_runtime_seconds"] = (
        merged["runtime_seconds_right"] - merged["runtime_seconds_left"]
    )
    ensure_dir(output_path.parent)
    merged.to_json(output_path, orient="records", indent=2)
    return output_path


def export_dataset_summary(dataset_path: Path, output_path: Path) -> Path:
    summary = describe_dataset(dataset_path)
    ensure_dir(output_path.parent)
    write_json(output_path, summary.model_dump(mode="json"))
    return output_path
