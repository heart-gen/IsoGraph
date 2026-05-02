"""Captum integrated gradients (Stage 8E — stub).

Requires the optional dependency group: pip install isograph[torch-explain]
"""

from __future__ import annotations


def compute_integrated_gradients(model, feature_table, eigengene, baseline=None):
    """Captum integrated gradients from feature_table → module eigengene.

    Returns DataFrame(feature_id, ig_score).
    """
    raise NotImplementedError("Stage 8E")
