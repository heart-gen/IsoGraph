"""Benchmark runner."""

from __future__ import annotations

import dataclasses
import json
import logging
import tracemalloc
from pathlib import Path
from time import perf_counter

import pandas as pd

from isograph.benchmarks.synthetic import (
    generate_core_suite,
    generate_multiplex_suite,
    generate_scale_suite,
)
from isograph.evaluation.metrics import (
    calibration_metrics,
    giant_component_fraction,
    module_recovery_score,
    role_aware_recall,
)
from isograph.evaluation.selection import stability_selection
from isograph.evaluation.snapshots import save_snapshot
from isograph.evaluation.tracking import tracking_run
from isograph.io.artifacts import describe_dataset, load_dataset_bundle
from isograph.io.real_data import freeze_real_dataset
from isograph.models.baseline import BaselineNetworkModel
from isograph.models.graph import GraphNetworkModel
from isograph.models.latent import LatentNetworkModel

try:
    from isograph.models.vae import VaeNetworkModel
    _VAE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VAE_AVAILABLE = False

from isograph.models.wgcna import WgcnaNetworkModel
from isograph.utils import ensure_dir, write_json
from isograph.workflow.config import BenchmarkCommandConfig

_SYNTHETIC_FIXTURES = frozenset({"toy_v1", "medium_v1", "realistic_v1", "realistic_unequal_v1"})
_SCALE_FIXTURES = frozenset({"xlarge_v1", "xxlarge_v1"})
_MULTIPLEX_FIXTURES = frozenset(
    {
        "toy_multiplex_v1",
        "medium_multiplex_v1",
        "noisy_multiplex_v1",
        "large_multiplex_v1",
        "xxlarge_multiplex_v1",
    }
)


def prepare_core_suite(config: BenchmarkCommandConfig) -> list[Path]:
    dataset_root = ensure_dir(config.dataset_root)
    f = config.fixture_filter
    paths: list[Path] = list(generate_core_suite(dataset_root, config.seed))
    if f not in _SYNTHETIC_FIXTURES:
        paths.append(freeze_real_dataset(config.real_data, dataset_root / config.dataset_suite))
    if f:
        paths = [p for p in paths if p.name == f]
    return paths


def prepare_scale_suite(config: BenchmarkCommandConfig) -> list[Path]:
    dataset_root = ensure_dir(config.dataset_root)
    f = config.fixture_filter
    paths: list[Path] = list(generate_scale_suite(dataset_root, config.seed))
    if f:
        paths = [p for p in paths if p.name == f]
    return paths


def prepare_multiplex_suite(config: BenchmarkCommandConfig) -> list[Path]:
    dataset_root = ensure_dir(config.dataset_root)
    f = config.fixture_filter
    paths: list[Path] = list(
        generate_multiplex_suite(
            dataset_root,
            config.seed,
            include_xxlarge=f == "xxlarge_multiplex_v1",
        )
    )
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
    elif config.backend == "vae":
        if not _VAE_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the vae backend. Install it with: pip install torch"
            )
        overrides = config.fixture_vae_overrides.get(dataset_name, {})
        vae_config = dataclasses.replace(config.vae, **overrides) if overrides else config.vae
        return VaeNetworkModel(vae_config), vae_config
    elif config.backend == "wgcna":
        overrides = config.fixture_wgcna_overrides.get(dataset_name, {})
        wgcna_config = dataclasses.replace(config.wgcna, **overrides) if overrides else config.wgcna
        return WgcnaNetworkModel(wgcna_config), wgcna_config
    else:
        overrides = config.fixture_model_overrides.get(dataset_name, {})
        model_config = dataclasses.replace(config.model, **overrides) if overrides else config.model
        return BaselineNetworkModel(model_config), model_config


def benchmark(config: BenchmarkCommandConfig) -> dict[str, Path]:
    if config.dataset_suite == "scale_v1":
        paths = prepare_scale_suite(config)
    elif config.dataset_suite == "multiplex_v1":
        paths = prepare_multiplex_suite(config)
    else:
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
                    gene_counts=bundle.matrices.get("gene_counts"),
                    gene_table=bundle.feature_tables.get("gene"),
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

            role_recall: dict | None = None
            giant_frac: float | None = None
            if config.dataset_suite == "multiplex_v1":
                truth_role = bundle.truth_tables.get("truth_channel_role.parquet", pd.DataFrame())
                if not truth_role.empty:
                    role_recall = role_aware_recall(artifacts.module_table, truth_role)
                n_genes = len(bundle.feature_tables.get("gene", pd.DataFrame()))
                giant_frac = giant_component_fraction(artifacts.edge_table, n_genes)

            # Stability selection: run only for real-data fixtures (no ground truth).
            recommended_alpha: float | None = None
            if config.run_stability_selection and truth.empty:
                sel_cfg = config.stability
                sel_result = stability_selection(
                    model=model,
                    transcript_counts=bundle.matrices["transcript_counts"],
                    transcript_table=bundle.feature_tables["transcript"],
                    sample_table=bundle.sample_table,
                    gene_counts=bundle.matrices.get("gene_counts"),
                    gene_table=bundle.feature_tables.get("gene"),
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

            selected_alpha_abundance: float | None = None
            if artifacts.calibration is not None:
                selected_alpha_abundance = artifacts.calibration.get("selected_alpha_abundance")

            report_row: dict = {
                "dataset": dataset_name,
                "n_samples": len(bundle.sample_table),
                "n_genes": len(bundle.feature_tables.get("gene", [])),
                "n_modules": n_modules,
                "n_edges": n_edges,
                "recovery": recovery,
                "runtime_seconds": elapsed,
                "recommended_alpha": recommended_alpha,
            }
            if role_recall is not None:
                report_row["role_recall"] = role_recall
            if giant_frac is not None:
                report_row["giant_component_fraction"] = giant_frac
            if selected_alpha_abundance is not None:
                report_row["selected_alpha_abundance"] = selected_alpha_abundance
            report_rows.append(report_row)
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
