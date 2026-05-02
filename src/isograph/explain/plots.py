"""Module explanation plots (Stage 8B — stub)."""

from __future__ import annotations


def plot_driver_bar(result, top_n: int = 20, ax=None):
    """Bar chart of top driver genes by |r|. Returns matplotlib Axes."""
    raise NotImplementedError("Stage 8B")


def plot_eigengene_heatmap(results, sample_meta=None, ax=None):
    """Heatmap of eigengenes × samples. Returns matplotlib Figure."""
    raise NotImplementedError("Stage 8B")


def plot_high_vs_low_violin(result, feature_ids=None, ax=None):
    """Violin/box of feature distributions by high/low module usage. Returns matplotlib Figure."""
    raise NotImplementedError("Stage 8B")
