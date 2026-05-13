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
    trait_columns: list[str] = field(default_factory=lambda: ["Age"])
    allow_abundance_abundance: bool = False
    alpha_switch: float | None = None
    alpha_abundance: float | None = None
    alpha_abundance_grid: list[float] | None = None


@dataclass
class BaselineModelConfig:
    name: str = "baseline_network"
    alpha: float = 0.12
    min_module_size: int = 3
    max_modules: int = 64
    residualize_covariates: list[str] = field(
        default_factory=lambda: ["RIN", "PMI", "mito_mapping_rate", "percent_assigned"]
    )
    trait_columns: list[str] = field(default_factory=lambda: ["Age"])
    allow_abundance_abundance: bool = False
    alpha_switch: float | None = None
    alpha_abundance: float | None = None
    alpha_abundance_grid: list[float] | None = None


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
    trait_columns: list[str] = field(default_factory=lambda: ["Age"])
    gamma: float = 0.5
    edge_types: list[str] = field(default_factory=lambda: ["corr"])
    corr_threshold: float = 0.3
    normalized_laplacian: bool = True
    allow_abundance_abundance: bool = False
    alpha_switch: float | None = None
    alpha_abundance: float | None = None
    alpha_abundance_grid: list[float] | None = None


@dataclass
class VaeModelConfig:
    """Configuration for the VAE network backend.

    **Choosing hidden_dim** (function of BOTH n_genes AND n_samples):

    - ``n_genes <= 1000`` and ``n_samples >= 150``: 128 (default) works well.
    - ``n_genes <= 1000`` and ``75 <= n_samples < 150``: raise to 192 for
      high-dispersion data.
    - ``n_genes <= 1000`` and ``n_samples < 75``: 256 reduces underfitting,
      but expect lower recovery; the model is data-limited.
    - ``1000 < n_genes <= 5000``: hidden_dim=256–512; consider n_hidden_layers=3.
    - ``n_genes > 5000`` (25:1–50:1 genes-to-samples): hidden_dim=512–1024;
      n_hidden_layers=3 recommended; batch_size will be auto-set to
      ``min(64, n_samples // 4)`` when left as None.

    Do not use a ratio like ``latent_dim * 8`` — stress tests show non-monotonic
    recovery across hidden_dim values, especially for high-dispersion or small-n data.

    **Choosing latent_dim / latent_dim_grid**:

    Set ``latent_dim_grid`` to a list (e.g. ``[2, 4, 6, 8, 12]``) to let the
    model sweep and auto-select the smallest *k* whose reconstruction RMSE
    improvement falls below 0.01. This is recommended when you do not know the
    number of true modules. Leave ``latent_dim_grid=None`` to use a fixed
    ``latent_dim``.
    """
    name: str = "vae_network"
    latent_dim: int = 8
    hidden_dim: int = 128
    n_hidden_layers: int = 2
    latent_dim_grid: list[int] | None = None
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
    device: str | None = None
    alpha: float = 0.70
    min_module_size: int = 2
    residualize_covariates: list[str] = field(
        default_factory=lambda: ["RIN", "PMI", "mito_mapping_rate", "percent_assigned"]
    )
    trait_columns: list[str] = field(default_factory=lambda: ["Age"])
    checkpoint_dir: Path | None = None
    allow_abundance_abundance: bool = False
    alpha_switch: float | None = None
    alpha_switch_grid: list[float] | None = None
    alpha_abundance: float | None = None
    alpha_abundance_grid: list[float] | None = None


@dataclass
class WgcnaModelConfig:
    name: str = "wgcna_network"
    power: int | None = None
    power_range: list[int] = field(default_factory=lambda: list(range(1, 21)))
    sft_r2_threshold: float = 0.85
    min_module_size: int = 2
    merge_cut_height: float = 0.25
    deep_split: int = 2
    network_type: str = "signed"
    random_state: int = 0
    timeout_seconds: int = 600
    trait_columns: list[str] = field(default_factory=lambda: ["Age"])
    residualize_covariates: list[str] = field(
        default_factory=lambda: ["RIN", "PMI", "mito_mapping_rate", "percent_assigned"]
    )


@dataclass
class RealDataFilterTerm:
    kind: str
    column: str
    df: int | None = None
    standardize: bool = True


@dataclass
class RealDataFreezeConfig:
    counts_root: Path = Path("data/counts")
    annotations_root: Path = Path("data/annotations")
    phenotype_tsv: Path = Path("data/phenotypes.tsv")
    ancestry_tsv: Path = Path("data/ancestry.txt")
    output_name: str = "real_caudate_aa_v1"
    gene_panel_size: int | None = 256
    allowed_diagnoses: list[str] = field(default_factory=lambda: ["Control", "SCZD"])
    cache_root: Path = Path("benchmarks/cache/real_data")
    filter_min_count: float = 10.0
    filter_min_total_count: float = 15.0
    filter_large_n: float = 10.0
    filter_min_prop: float = 0.7
    filter_design_terms: list[RealDataFilterTerm] = field(
        default_factory=lambda: [RealDataFilterTerm(kind="natural_spline", column="Age", df=3)]
    )


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
    backend: str = "vae"  # "baseline" | "latent" | "graph" | "vae" | "wgcna"
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
    wgcna: WgcnaModelConfig = field(default_factory=WgcnaModelConfig)
    # Per-fixture WGCNA config overrides for the wgcna backend.
    fixture_wgcna_overrides: dict[str, dict] = field(default_factory=dict)
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
    backend: str = "vae"  # "baseline" | "latent" | "graph" | "vae" | "wgcna"
    tracking_uri: str | None = None
    seed: int = 7
    model: BaselineModelConfig = field(default_factory=BaselineModelConfig)
    latent: LatentModelConfig = field(default_factory=LatentModelConfig)
    graph: GraphModelConfig = field(default_factory=GraphModelConfig)
    vae: VaeModelConfig = field(default_factory=VaeModelConfig)
    wgcna: WgcnaModelConfig = field(default_factory=WgcnaModelConfig)


@dataclass
class CompareCommandConfig:
    command: str = "compare"
    reference: Path | None = None
    candidate: Path | None = None
    output_path: Path = Path("artifacts/reports/comparison.json")


@dataclass
class ExplainCommandConfig:
    command: str = "explain-module"
    artifact_dir: Path = Path("artifacts/fits/manual")
    feature_table_path: Path | None = None
    feature_meta_path: Path | None = None
    module_ids: list[str] | None = None
    output_dir: Path = Path("artifacts/explain")
    module_score_table_path: Path | None = None
    split_percentile: float = 50.0
    min_complete_pairs: int = 3
    fdr_method: str = "bh"
    transcript_usage_feature_type: str = "transcript_usage"
