"""Model interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FitArtifacts:
    module_table: pd.DataFrame
    edge_table: pd.DataFrame
    trait_table: pd.DataFrame
    feature_scores: pd.DataFrame


class NetworkModel:
    def fit(self, *args, **kwargs) -> FitArtifacts:
        raise NotImplementedError
