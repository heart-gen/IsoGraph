"""Benchmark runner."""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from time import perf_counter

import pandas as pd

from isograph.benchmarks.synthetic import generate_core_suite
from isograph.evaluation.metrics import module_recovery_score
from isograph.evaluation.snapshots import save_snapshot
from isograph.evaluation.tracking import tracking_run
from isograph.io.artifacts import describe_dataset, load_dataset_bundle
from isograph.io.real_data import freeze_real_dataset
from isograph.models.baseline import BaselineNetworkModel
from isograph.utils import ensure_dir, write_json
from isograph.workflow.config import BenchmarkCommandConfig

_SYNTHETIC_FIXTURES = frozenset({"toy_v1", "medium_v1"})


def prepare_core_suite(config: BenchmarkCommandConfig) -> list[Path]:
    dataset_root = ensure_dir(config.dataset_root)
    f = config.fixture_filter
    paths: list[Path] = list(generate_core_suite(dataset_root, config.seed))
    if f not in _SYNTHETIC_FIXTURES:
        paths.append(freeze_real_dataset(config.real_data, dataset_root / config.dataset_suite))
    if f:
        paths = [p for p in paths if p.name == f]
    return paths


def benchmark(config: BenchmarkCommandConfig) -> dict[str, Path]:
    paths = prepare_core_suite(config)
    report_rows = []
    rt_rows = []
    model = BaselineNetworkModel(config.model)
    stage_artifacts_dir = ensure_dir(config.artifacts_root / "benchmarks" / config.stage_name)

    with tracking_run(config.tracking_uri, run_name=f"{config.dataset_suite}-{config.model.name}"):
        for dataset_dir in paths:
            bundle = load_dataset_bundle(dataset_dir)
            dataset_name = bundle.manifest.dataset_name

            tracemalloc.start()
            start = perf_counter()
            artifacts = model.fit(
                transcript_counts=bundle.matrices["transcript_counts"],
                transcript_table=bundle.feature_tables["transcript"],
                sample_table=bundle.sample_table,
            )
            elapsed = perf_counter() - start
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
            recovery = module_recovery_score(artifacts.module_table, truth) if not truth.empty else None
            n_modules = 0 if artifacts.module_table.empty else artifacts.module_table["module_id"].nunique()
            n_edges = len(artifacts.edge_table)

            metrics = {
                "n_modules": n_modules,
                "n_edges": n_edges,
                "recovery": recovery,
                "runtime_seconds": elapsed,
            }
            save_snapshot(
                fit_artifacts=artifacts,
                model_config=config.model,
                metrics=metrics,
                output_dir=ensure_dir(stage_artifacts_dir / dataset_name),
                snapshot_name=f"stage1_{dataset_name}_baseline_v1_seed{config.seed:04d}",
                dataset_name=dataset_name,
            )

            report_rows.append(
                {
                    "dataset": dataset_name,
                    "n_samples": len(bundle.sample_table),
                    "n_genes": len(bundle.feature_tables.get("gene", [])),
                    "n_modules": n_modules,
                    "n_edges": n_edges,
                    "recovery": recovery,
                    "runtime_seconds": elapsed,
                }
            )
            rt_rows.append(
                {
                    "dataset": dataset_name,
                    "runtime_seconds": elapsed,
                    "peak_memory_bytes": peak_bytes,
                }
            )

    gate_failures = [
        f"{row['dataset']}: recovery={row['recovery']:.4f} < threshold={config.recovery_thresholds[row['dataset']]}"
        for row in report_rows
        if row["dataset"] in config.recovery_thresholds
        and row["recovery"] is not None
        and row["recovery"] < config.recovery_thresholds[row["dataset"]]
    ]

    report_dir = ensure_dir(config.report_root)
    report_path = report_dir / f"{config.stage_name}-benchmark.json"
    write_json(
        report_path,
        {
            "results": report_rows,
            "suite": config.dataset_suite,
            "model": config.model.name,
            "stage": config.stage_name,
            "gate_failures": gate_failures,
        },
    )

    rt_path = report_dir / f"{config.stage_name}-runtime-memory.json"
    write_json(
        rt_path,
        {"results": rt_rows, "suite": config.dataset_suite, "stage": config.stage_name},
    )

    return {"report": report_path, "runtime_memory": rt_path}


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
