"""Benchmark runner."""

from __future__ import annotations

import dataclasses
import json
import logging
import tracemalloc
from pathlib import Path
from time import perf_counter

import pandas as pd

from isograph.benchmarks.synthetic import generate_core_suite
from isograph.evaluation.metrics import calibration_metrics, module_recovery_score
from isograph.evaluation.selection import stability_selection
from isograph.evaluation.snapshots import save_snapshot
from isograph.evaluation.tracking import tracking_run
from isograph.io.artifacts import describe_dataset, load_dataset_bundle
from isograph.io.real_data import freeze_real_dataset
from isograph.models.baseline import BaselineNetworkModel
from isograph.models.graph import GraphNetworkModel
from isograph.models.latent import LatentNetworkModel
from isograph.utils import ensure_dir, write_json
from isograph.workflow.config import BenchmarkCommandConfig

_SYNTHETIC_FIXTURES = frozenset({"toy_v1", "medium_v1", "realistic_v1", "realistic_unequal_v1"})


def prepare_core_suite(config: BenchmarkCommandConfig) -> list[Path]:
    dataset_root = ensure_dir(config.dataset_root)
    f = config.fixture_filter
    paths: list[Path] = list(generate_core_suite(dataset_root, config.seed))
    if f not in _SYNTHETIC_FIXTURES:
        paths.append(freeze_real_dataset(config.real_data, dataset_root / config.dataset_suite))
    if f:
        paths = [p for p in paths if p.name == f]
    return paths


def _make_model(config: BenchmarkCommandConfig, dataset_name: str):
    if config.backend == "graph":
        overrides = config.fixture_graph_overrides.get(dataset_name, {})
        graph_config = dataclasses.replace(config.graph, **overrides) if overrides else config.graph
        return GraphNetworkModel(graph_config), graph_config
    elif config.backend == "latent":
        overrides = config.fixture_latent_overrides.get(dataset_name, {})
        latent_config = dataclasses.replace(config.latent, **overrides) if overrides else config.latent
        return LatentNetworkModel(latent_config), latent_config
    else:
        overrides = config.fixture_model_overrides.get(dataset_name, {})
        model_config = dataclasses.replace(config.model, **overrides) if overrides else config.model
        return BaselineNetworkModel(model_config), model_config


def benchmark(config: BenchmarkCommandConfig) -> dict[str, Path]:
    paths = prepare_core_suite(config)
    report_rows = []
    rt_rows = []
    artifacts_by_fixture = {}
    stage_artifacts_dir = ensure_dir(config.artifacts_root / "benchmarks" / config.stage_name)

    with tracking_run(config.tracking_uri, run_name=f"{config.dataset_suite}-{config.backend}"):
        for dataset_dir in paths:
            bundle = load_dataset_bundle(dataset_dir)
            dataset_name = bundle.manifest.dataset_name

            model, model_config = _make_model(config, dataset_name)

            fixture_artifact_dir = ensure_dir(stage_artifacts_dir / dataset_name)
            log_path = fixture_artifact_dir / "run.log"
            isograph_logger = logging.getLogger("isograph")
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            isograph_logger.addHandler(file_handler)

            tracemalloc.start()
            start = perf_counter()
            try:
                artifacts = model.fit(
                    transcript_counts=bundle.matrices["transcript_counts"],
                    transcript_table=bundle.feature_tables["transcript"],
                    sample_table=bundle.sample_table,
                )
            finally:
                elapsed = perf_counter() - start
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                isograph_logger.removeHandler(file_handler)
                file_handler.close()

            artifacts_by_fixture[dataset_name] = artifacts

            truth = bundle.truth_tables.get("truth_modules.parquet", pd.DataFrame())
            recovery = module_recovery_score(artifacts.module_table, truth) if not truth.empty else None
            n_modules = 0 if artifacts.module_table.empty else artifacts.module_table["module_id"].nunique()
            n_edges = len(artifacts.edge_table)

            # Stability selection: run only for real-data fixtures (no ground truth).
            recommended_alpha: float | None = None
            if config.run_stability_selection and truth.empty:
                sel_cfg = config.stability
                sel_result = stability_selection(
                    model=model,
                    transcript_counts=bundle.matrices["transcript_counts"],
                    transcript_table=bundle.feature_tables["transcript"],
                    sample_table=bundle.sample_table,
                    alpha_grid=sel_cfg.alpha_grid,
                    n_iterations=sel_cfg.n_iterations,
                    subsample_fraction=sel_cfg.subsample_fraction,
                    stability_threshold=sel_cfg.stability_threshold,
                    seed=sel_cfg.seed,
                )
                recommended_alpha = sel_result.recommended_alpha
                sel_path = ensure_dir(config.report_root) / f"{config.stage_name}-{dataset_name}-stability.json"
                write_json(
                    sel_path,
                    {
                        "dataset": dataset_name,
                        "stage": config.stage_name,
                        "backend": config.backend,
                        "recommended_alpha": recommended_alpha,
                        "summary": sel_result.summary_table().to_dict(orient="records"),
                    },
                )

            metrics = {
                "n_modules": n_modules,
                "n_edges": n_edges,
                "recovery": recovery,
                "runtime_seconds": elapsed,
            }
            snap_name = f"{config.stage_name}_{dataset_name}_{config.backend}_v1_seed{config.seed:04d}"
            save_snapshot(
                fit_artifacts=artifacts,
                model_config=model_config,
                metrics=metrics,
                output_dir=ensure_dir(stage_artifacts_dir / dataset_name),
                snapshot_name=snap_name,
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
                    "recommended_alpha": recommended_alpha,
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
            "backend": config.backend,
            "stage": config.stage_name,
            "gate_failures": gate_failures,
        },
    )

    rt_path = report_dir / f"{config.stage_name}-runtime-memory.json"
    write_json(
        rt_path,
        {"results": rt_rows, "suite": config.dataset_suite, "stage": config.stage_name},
    )

    result: dict[str, Path] = {"report": report_path, "runtime_memory": rt_path}

    # Write calibration report when any fixture produced calibration data.
    cal = calibration_metrics(artifacts_by_fixture)
    if cal["calibration_by_fixture"]:
        cal_path = report_dir / f"{config.stage_name}-calibration.json"
        write_json(cal_path, {**cal, "stage": config.stage_name, "backend": config.backend})
        result["calibration"] = cal_path

    return result


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
