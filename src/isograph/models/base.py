"""Model interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class FitArtifacts:
    module_table: pd.DataFrame
    edge_table: pd.DataFrame
    trait_table: pd.DataFrame
    feature_scores: pd.DataFrame
    calibration: dict | None = None
    checkpoint_path: Path | None = None
    eigengene_table: pd.DataFrame | None = None


class NetworkModel:
    def fit(self, *args, **kwargs) -> FitArtifacts:
        raise NotImplementedError
