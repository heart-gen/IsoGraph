"""Configuration and result types for module explanation (Stage 7A)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ExplainConfig:
    split_percentile: float = 50.0
    min_complete_pairs: int = 3
    fdr_method: str = "bh"
    transcript_usage_feature_type: str = "transcript_usage"
    plot: bool = False
    output_format: str | list[str] = "png"


@dataclass
class ExplainResult:
    module_id: str
    gene_driver_table: pd.DataFrame
    transcript_polarity_table: pd.DataFrame
    high_vs_low_table: pd.DataFrame
    eigengene: np.ndarray
    n_module_genes: int
    sample_ids: list[str] = field(default_factory=list)
