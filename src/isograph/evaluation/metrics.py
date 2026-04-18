"""Benchmark metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from isograph.models.base import FitArtifacts


def calibration_metrics(artifacts_by_fixture: dict[str, "FitArtifacts"]) -> dict:
    """Aggregate calibration dicts from fit artifacts keyed by fixture name."""
    rows = []
    for fixture_name, arts in artifacts_by_fixture.items():
        if arts.calibration is not None:
            rows.append({"fixture": fixture_name, **arts.calibration})
    return {"calibration_by_fixture": rows}


def module_recovery_score(predicted: pd.DataFrame, truth: pd.DataFrame) -> float:
    if predicted.empty or truth.empty:
        return 0.0
    predicted_groups = [set(group["gene_id"]) for _, group in predicted.groupby("module_id")]
    truth_groups = [set(group["gene_id"]) for _, group in truth.groupby("module_id")]
    total = 0.0
    for truth_group in truth_groups:
        best = 0.0
        for predicted_group in predicted_groups:
            union = len(truth_group | predicted_group)
            if union == 0:
                continue
            best = max(best, len(truth_group & predicted_group) / union)
        total += best
    return total / len(truth_groups)
