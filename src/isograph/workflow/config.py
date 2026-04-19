"""Typed workflow configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LatentModelConfig:
    name: str = "latent_network"
    n_components: int = 10
    # When set, sweep over this grid and pick k by cross-validated log-likelihood.
    # Overrides n_components. Use consecutive integers through the likely signal
    # range so no true k is skipped, then sparse above.
    n_components_grid: list[int] | None = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    )
    n_components_cv_folds: int = 5
    alpha: float = 0.10
    min_module_size: int = 2
    max_iter: int = 1000
    tol: float = 1e-4
    residualize_covariates: list[str] = field(
        default_factory=lambda: ["RIN", "PMI", "mito_mapping_rate", "percent_assigned"]
    )
    trait_columns: list[str] = field(default_factory=lambda: ["Dx", "Age"])


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
class GraphModelConfig:
    name: str = "graph_network"
    n_components: int = 10
    n_components_grid: list[int] | None = field(
        default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    )
    n_components_cv_folds: int = 5
    alpha: float = 0.10
    min_module_size: int = 2
    max_iter: int = 1000
    tol: float = 1e-4
    residualize_covariates: list[str] = field(
        default_factory=lambda: ["RIN", "PMI", "mito_mapping_rate", "percent_assigned"]
    )
    trait_columns: list[str] = field(default_factory=lambda: ["Dx", "Age"])
    gamma: float = 0.5
    edge_types: list[str] = field(default_factory=lambda: ["corr"])
    corr_threshold: float = 0.3
    normalized_laplacian: bool = True


@dataclass
class VaeModelConfig:
    name: str = "vae_network"
    latent_dim: int = 8
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    beta: float = 1.0
    n_epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int | None = None
    warmup_epochs: int | None = None
    val_fraction: float = 0.2
    patience: int = 50
    early_stop_tol: float = 1e-4
    collapse_threshold: float = 0.01
    random_state: int = 0
    alpha: float = 0.70
    min_module_size: int = 2
    residualize_covariates: list[str] = field(
        default_factory=lambda: ["RIN", "PMI", "mito_mapping_rate", "percent_assigned"]
    )
    trait_columns: list[str] = field(default_factory=lambda: ["Dx", "Age"])
    checkpoint_dir: Path | None = None


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
class StabilitySelectionConfig:
    """Config for stability-selection alpha tuning (for real data without ground truth)."""
    alpha_grid: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    )
    n_iterations: int = 50
    subsample_fraction: float = 0.8
    stability_threshold: float = 0.6
    seed: int = 0


@dataclass
class BenchmarkCommandConfig:
    command: str = "benchmark"
    dataset_suite: str = "core_v1"
    stage_name: str = "stage1_baseline"
    backend: str = "baseline"  # "baseline" | "latent" | "graph" | "vae"
    fixture_filter: str | None = None  # None = run all; "toy_v1"|"medium_v1"|"real_caudate_aa_v1" = one fixture
    benchmark_root: Path = Path("benchmarks")
    artifacts_root: Path = Path("artifacts")
    dataset_root: Path = Path("benchmarks/datasets")
    report_root: Path = Path("artifacts/reports")
    tracking_uri: str | None = None
    seed: int = 7
    real_data: RealDataFreezeConfig = field(default_factory=RealDataFreezeConfig)
    model: BaselineModelConfig = field(default_factory=BaselineModelConfig)
    latent: LatentModelConfig = field(default_factory=LatentModelConfig)
    graph: GraphModelConfig = field(default_factory=GraphModelConfig)
    recovery_thresholds: dict[str, float] = field(
        default_factory=lambda: {"toy_v1": 1.0, "medium_v1": 0.875}
    )
    # Per-fixture model config overrides (e.g. different alpha per fixture).
    # Keys are fixture names; values are partial BaselineModelConfig field dicts.
    fixture_model_overrides: dict[str, dict] = field(default_factory=dict)
    # Per-fixture latent config overrides for the latent backend.
    fixture_latent_overrides: dict[str, dict] = field(default_factory=dict)
    # Per-fixture graph config overrides for the graph backend.
    fixture_graph_overrides: dict[str, dict] = field(default_factory=dict)
    vae: VaeModelConfig = field(default_factory=VaeModelConfig)
    # Per-fixture VAE config overrides for the vae backend.
    fixture_vae_overrides: dict[str, dict] = field(default_factory=dict)
    # When True, run stability selection on real-data fixtures and append
    # recommended_alpha to the benchmark report for each such fixture.
    run_stability_selection: bool = False
    stability: StabilitySelectionConfig = field(default_factory=StabilitySelectionConfig)


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
