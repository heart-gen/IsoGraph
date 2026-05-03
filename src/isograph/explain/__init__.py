"""Module explanation (Stage 7A / 8B / 8C / 8D)."""

from __future__ import annotations

from isograph.explain.annotation import (
    annotate_driver_table,
    annotate_transcript_table,
    load_annotation_table,
)
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
from isograph.explain.structure import annotate_switch_pairs
from isograph.explain.vae_attribution import compute_decoder_jacobian, filter_vae_drivers

__all__ = [
    "ExplainConfig",
    "ExplainResult",
    "explain_module",
    "annotate_switch_pairs",
    "load_annotation_table",
    "annotate_driver_table",
    "annotate_transcript_table",
    "plot_driver_bar",
    "plot_eigengene_heatmap",
    "plot_high_vs_low_violin",
    "plot_isoform_gradient",
    "plot_summary_panel",
    "plot_switch_pair",
    "plot_transcript_polarity_heatmap",
    "summarize_module",
    "compute_decoder_jacobian",
    "filter_vae_drivers",
]
