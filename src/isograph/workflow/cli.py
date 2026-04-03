"""CLI entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from isograph.evaluation.runner import benchmark, compare_reports, export_dataset_summary
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

    subparsers.add_parser("benchmark")
    freeze = subparsers.add_parser("freeze-real")
    freeze.add_argument("--suite-name", default="core_v1")

    fit = subparsers.add_parser("fit")
    fit.add_argument("--dataset-path", required=True)
    fit.add_argument("--output-dir", default="artifacts/fits/manual")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--left-report", required=True)
    compare.add_argument("--right-report", required=True)
    compare.add_argument("--output-path", default="artifacts/reports/comparison.json")

    export = subparsers.add_parser("export")
    export.add_argument("--dataset-path", required=True)
    export.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    cli_args, overrides = _split_overrides(argv)
    parser = build_parser()
    args = parser.parse_args(cli_args)

    if args.command == "benchmark":
        payload = load_config("benchmark", overrides)
        config = instantiate_dataclass(BenchmarkCommandConfig, payload)
        output = benchmark(config)
        print(output)
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
        config = instantiate_dataclass(FitCommandConfig, payload)
        bundle = load_dataset_bundle(Path(config.dataset_path))
        artifacts = BaselineNetworkModel(config.model).fit(
            transcript_counts=bundle.matrices["transcript_counts"],
            transcript_table=bundle.feature_tables["transcript"],
            sample_table=bundle.sample_table,
        )
        output_dir = ensure_dir(Path(config.output_dir))
        artifacts.module_table.to_parquet(output_dir / "modules.parquet", index=False)
        artifacts.edge_table.to_parquet(output_dir / "edges.parquet", index=False)
        artifacts.trait_table.to_parquet(output_dir / "traits.parquet", index=False)
        artifacts.feature_scores.to_parquet(output_dir / "feature_scores.parquet", index=False)
        write_json(output_dir / "fit_config.json", dataclass_to_jsonable(config))
        print(output_dir)
        return

    if args.command == "compare":
        payload = load_config("compare", overrides)
        payload["left_report"] = args.left_report
        payload["right_report"] = args.right_report
        payload["output_path"] = args.output_path
        config = instantiate_dataclass(CompareCommandConfig, payload)
        output = compare_reports(Path(config.left_report), Path(config.right_report), Path(config.output_path))
        print(output)
        return

    if args.command == "export":
        output = export_dataset_summary(Path(args.dataset_path), Path(args.output_path))
        print(output)
