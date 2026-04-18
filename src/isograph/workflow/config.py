"""Typed workflow configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaselineModelConfig:
    name: str = "baseline_network"
    alpha: float = 0.12
    min_module_size: int = 3
    max_modules: int = 64
    residualize_covariates: list[str] = field(
        default_factory=lambda: ["RIN", "PMI", "mito_mapping_rate", "percent_assigned"]
    )
    trait_columns: list[str] = field(default_factory=lambda: ["Dx", "Age"])


@dataclass
class RealDataFreezeConfig:
    counts_root: Path = Path("data/counts")
    annotations_root: Path = Path("data/annotations")
    phenotype_tsv: Path = Path("data/phenotypes.tsv")
    ancestry_tsv: Path = Path("data/ancestry.txt")
    output_name: str = "real_caudate_aa_v1"
    gene_panel_size: int = 256
    allowed_diagnoses: list[str] = field(default_factory=lambda: ["Control", "SCZD"])
    cache_root: Path = Path("benchmarks/cache/real_data")


@dataclass
class BenchmarkCommandConfig:
    command: str = "benchmark"
    dataset_suite: str = "core_v1"
    benchmark_root: Path = Path("benchmarks")
    artifacts_root: Path = Path("artifacts")
    dataset_root: Path = Path("benchmarks/datasets")
    report_root: Path = Path("artifacts/reports")
    tracking_uri: str | None = None
    seed: int = 7
    real_data: RealDataFreezeConfig = field(default_factory=RealDataFreezeConfig)
    model: BaselineModelConfig = field(default_factory=BaselineModelConfig)


@dataclass
class FitCommandConfig:
    command: str = "fit"
    benchmark_root: Path = Path("benchmarks")
    artifacts_root: Path = Path("artifacts")
    dataset_path: Path | None = None
    output_dir: Path = Path("artifacts/fits/manual")
    tracking_uri: str | None = None
    seed: int = 7
    model: BaselineModelConfig = field(default_factory=BaselineModelConfig)


@dataclass
class CompareCommandConfig:
    command: str = "compare"
    reference: Path | None = None
    candidate: Path | None = None
    output_path: Path = Path("artifacts/reports/comparison.json")
