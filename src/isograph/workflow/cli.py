"""CLI entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from isograph.evaluation.runner import benchmark, compare_reports, export_dataset_summary
from isograph.evaluation.snapshots import compare_snapshot_dirs
from isograph.io.real_data import freeze_real_dataset
from isograph.models.baseline import BaselineNetworkModel
from isograph.io.artifacts import load_dataset_bundle
from isograph.utils import dataclass_to_jsonable, ensure_dir, write_json
from isograph.workflow.config import BenchmarkCommandConfig, CompareCommandConfig, FitCommandConfig
from isograph.workflow.hydra import instantiate_dataclass, load_config


def _split_overrides(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    marker = argv.index("--")
    return argv[:marker], argv[marker + 1 :]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="isograph")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark")
    bench.add_argument("--config-name", default="benchmark", help="Hydra config name (without .yaml)")
    freeze = subparsers.add_parser("freeze-real")
    freeze.add_argument("--suite-name", default="core_v1")

    fit = subparsers.add_parser("fit")
    fit.add_argument("--dataset-path", required=True)
    fit.add_argument("--output-dir", default="artifacts/fits/manual")
    fit.add_argument(
        "--backend",
        default=None,
        choices=["baseline", "latent", "graph", "vae", "wgcna"],
        help="Network model backend (default: value in fit.yaml, usually 'baseline')",
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", required=True, help="Reference snapshot directory or benchmark JSON report")
    compare.add_argument("--candidate", required=True, help="Candidate snapshot directory or benchmark JSON report")
    compare.add_argument("--output-path", default="artifacts/reports/comparison.json")

    export = subparsers.add_parser("export")
    export.add_argument("--dataset-path", required=True)
    export.add_argument("--output-path", required=True)

    explain = subparsers.add_parser("explain-module")
    explain.add_argument("--artifact-dir", required=True, dest="artifact_dir")
    explain.add_argument("--feature-table", required=True, dest="feature_table")
    explain.add_argument("--feature-meta", required=True, dest="feature_meta")
    explain.add_argument(
        "--module-ids", nargs="*", default=None, dest="module_ids",
        help="Module IDs to explain (default: all). Example: --module-ids M000 M001",
    )
    explain.add_argument("--output-dir", default="artifacts/explain", dest="output_dir")
    explain.add_argument("--module-score-table", default=None, dest="module_score_table")
    explain.add_argument("--split-percentile", type=float, default=50.0)
    explain.add_argument("--min-complete-pairs", type=int, default=3)
    explain.add_argument("--fdr-method", default="bh")
    explain.add_argument(
        "--plot", action="store_true", default=False,
        help="Write plot files alongside parquet outputs.",
    )
    explain.add_argument(
        "--output-format", nargs="+", default=["png"],
        choices=["png", "pdf"],
        dest="output_format",
        help="Plot output format(s): png, pdf, or both (default: png).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    cli_args, overrides = _split_overrides(argv)
    parser = build_parser()
    args = parser.parse_args(cli_args)

    if args.command == "benchmark":
        payload = load_config(args.config_name, overrides)
        config = instantiate_dataclass(BenchmarkCommandConfig, payload)
        output = benchmark(config)
        print(output["report"])
        print(output["runtime_memory"])
        return

    if args.command == "freeze-real":
        payload = load_config("benchmark", overrides)
        config = instantiate_dataclass(BenchmarkCommandConfig, payload)
        suite_dir = ensure_dir(config.dataset_root / args.suite_name)
        output = freeze_real_dataset(config.real_data, suite_dir)
        print(output)
        return

    if args.command == "fit":
        payload = load_config("fit", overrides)
        payload["dataset_path"] = args.dataset_path
        payload["output_dir"] = args.output_dir
        if args.backend is not None:
            payload["backend"] = args.backend
        config = instantiate_dataclass(FitCommandConfig, payload)
        bundle = load_dataset_bundle(Path(config.dataset_path))
        fit_kwargs = dict(
            transcript_counts=bundle.matrices["transcript_counts"],
            transcript_table=bundle.feature_tables["transcript"],
            sample_table=bundle.sample_table,
        )
        if config.backend == "baseline":
            model = BaselineNetworkModel(config.model)
        elif config.backend == "latent":
            from isograph.models.latent import LatentNetworkModel
            model = LatentNetworkModel(config.latent)
        elif config.backend == "graph":
            from isograph.models.graph import GraphNetworkModel
            model = GraphNetworkModel(config.graph)
        elif config.backend == "vae":
            from isograph.models.vae import VaeNetworkModel
            model = VaeNetworkModel(config.vae)
        elif config.backend == "wgcna":
            from isograph.models.wgcna import WgcnaNetworkModel
            model = WgcnaNetworkModel(config.wgcna)
        else:
            raise ValueError(f"Unknown backend: {config.backend!r}")
        artifacts = model.fit(**fit_kwargs)
        output_dir = ensure_dir(Path(config.output_dir))
        artifacts.module_table.to_parquet(output_dir / "modules.parquet", index=False)
        artifacts.edge_table.to_parquet(output_dir / "edges.parquet", index=False)
        artifacts.trait_table.to_parquet(output_dir / "traits.parquet", index=False)
        artifacts.feature_scores.to_parquet(output_dir / "feature_scores.parquet", index=False)
        write_json(output_dir / "fit_config.json", dataclass_to_jsonable(config))
        if artifacts.calibration:
            write_json(output_dir / "calibration.json", artifacts.calibration)
        print(output_dir)
        return

    if args.command == "compare":
        reference = Path(args.reference)
        candidate = Path(args.candidate)
        output_path = Path(args.output_path)
        if reference.is_dir() and candidate.is_dir():
            # Snapshot comparison: reference/candidate are artifact directories.
            ensure_dir(output_path.parent)
            report = compare_snapshot_dirs(reference, candidate)
            write_json(output_path, report)
            print(output_path)
        else:
            # Legacy benchmark-report comparison: reference/candidate are JSON files.
            output = compare_reports(reference, candidate, output_path)
            print(output)
        return

    if args.command == "export":
        output = export_dataset_summary(Path(args.dataset_path), Path(args.output_path))
        print(output)
        return

    if args.command == "explain-module":
        import pandas as pd
        from isograph.explain.config import ExplainConfig
        from isograph.explain.core import explain_module

        feature_table = pd.read_parquet(args.feature_table)
        feature_meta = pd.read_parquet(args.feature_meta)
        module_score_table = (
            pd.read_parquet(args.module_score_table)
            if args.module_score_table is not None
            else None
        )
        output_format = args.output_format[0] if len(args.output_format) == 1 else args.output_format
        config = ExplainConfig(
            split_percentile=args.split_percentile,
            min_complete_pairs=args.min_complete_pairs,
            fdr_method=args.fdr_method,
            plot=args.plot,
            output_format=output_format,
        )
        explain_module(
            artifact_dir=Path(args.artifact_dir),
            feature_table=feature_table,
            feature_meta=feature_meta,
            module_ids=args.module_ids if args.module_ids else None,
            output_dir=Path(args.output_dir),
            module_score_table=module_score_table,
            config=config,
        )
        print(Path(args.output_dir) / "module_explanation_manifest.json")
