"""Benchmark metrics."""

from __future__ import annotations

from itertools import product

import pandas as pd


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
