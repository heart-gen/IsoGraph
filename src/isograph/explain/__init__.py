"""Module explanation (Stage 7A / 8B)."""

from __future__ import annotations

from isograph.explain.config import ExplainConfig, ExplainResult
from isograph.explain.core import explain_module
from isograph.explain.plots import (
    plot_driver_bar,
    plot_eigengene_heatmap,
    plot_high_vs_low_violin,
    plot_isoform_gradient,
    plot_summary_panel,
    plot_switch_pair,
    plot_transcript_polarity_heatmap,
    summarize_module,
)

__all__ = [
    "ExplainConfig",
    "ExplainResult",
    "explain_module",
    "plot_driver_bar",
    "plot_eigengene_heatmap",
    "plot_high_vs_low_violin",
    "plot_isoform_gradient",
    "plot_summary_panel",
    "plot_switch_pair",
    "plot_transcript_polarity_heatmap",
    "summarize_module",
]
