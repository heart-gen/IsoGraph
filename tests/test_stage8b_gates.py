"""Stage 8B gate tests — publication-ready explanation plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from isograph.explain.config import ExplainConfig, ExplainResult
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


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_result(n_genes: int = 10, n_samples: int = 20, seed: int = 0, module_id: str = "M000") -> ExplainResult:
    rng = np.random.default_rng(seed)
    gene_ids = [f"G{i:04d}" for i in range(n_genes)]
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    r_vals = rng.uniform(-1, 1, size=n_genes)

    gene_driver_table = pd.DataFrame({
        "gene_id": gene_ids,
        "r": r_vals,
        "pvalue": np.abs(r_vals) * 0.1,
        "qvalue": np.abs(r_vals) * 0.15,
        "n_samples": [n_samples] * n_genes,
        "missing_fraction": [0.0] * n_genes,
    }).sort_values("r", key=np.abs, ascending=False).reset_index(drop=True)

    feature_ids = [f"{g}_T0" for g in gene_ids]
    mean_h = rng.uniform(0.3, 0.8, n_genes)
    mean_l = rng.uniform(0.2, 0.6, n_genes)
    delta = mean_h - mean_l
    se = rng.uniform(0.02, 0.1, n_genes)

    high_vs_low_table = pd.DataFrame({
        "feature_id": feature_ids,
        "gene_id": gene_ids,
        "mean_high": mean_h,
        "mean_low": mean_l,
        "delta": delta,
        "se": se,
        "tstat": delta / np.maximum(se, 1e-9),
        "pvalue": rng.uniform(0, 0.1, n_genes),
        "n_high": [n_samples // 2] * n_genes,
        "n_low": [n_samples // 2] * n_genes,
        "missing_fraction": [0.0] * n_genes,
    })

    eigengene = rng.normal(0, 1, n_samples)
    return ExplainResult(
        module_id=module_id,
        gene_driver_table=gene_driver_table,
        transcript_polarity_table=pd.DataFrame(),
        high_vs_low_table=high_vs_low_table,
        eigengene=eigengene,
        n_module_genes=n_genes,
        sample_ids=sample_ids,
    )


def _make_result_with_transcripts(n_genes: int = 8, n_tx_per_gene: int = 3,
                                   n_samples: int = 20, seed: int = 1,
                                   module_id: str = "M000") -> ExplainResult:
    """ExplainResult with a populated transcript_polarity_table."""
    rng = np.random.default_rng(seed)
    gene_ids = [f"G{i:04d}" for i in range(n_genes)]
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    r_vals = rng.uniform(-0.9, 0.9, size=n_genes)

    gene_driver_table = pd.DataFrame({
        "gene_id": gene_ids,
        "r": r_vals,
        "pvalue": np.abs(r_vals) * 0.1,
        "qvalue": np.abs(r_vals) * 0.15,
        "n_samples": [n_samples] * n_genes,
        "missing_fraction": [0.0] * n_genes,
    }).sort_values("r", key=np.abs, ascending=False).reset_index(drop=True)

    tx_rows = []
    for gid in gene_ids:
        tx_rs = rng.uniform(-1, 1, size=n_tx_per_gene)
        for i, r in enumerate(tx_rs):
            fid = f"{gid}_T{i}"
            tx_rows.append({
                "feature_id": fid, "gene_id": gid, "r": float(r),
                "pvalue": abs(r) * 0.1, "qvalue": abs(r) * 0.15,
                "n_samples": n_samples, "missing_fraction": 0.0,
            })
    tx_df = pd.DataFrame(tx_rows)
    # add switch_strength = max(r) - min(r) per gene
    ss = tx_df.groupby("gene_id")["r"].agg(lambda x: x.max() - x.min()).rename("switch_strength")
    tx_df = tx_df.merge(ss, on="gene_id")

    feature_ids = tx_df["feature_id"].tolist()
    mean_h = rng.uniform(0.3, 0.8, len(feature_ids))
    mean_l = rng.uniform(0.1, 0.5, len(feature_ids))
    delta = mean_h - mean_l
    se = rng.uniform(0.02, 0.1, len(feature_ids))
    hvl_table = pd.DataFrame({
        "feature_id": feature_ids,
        "gene_id": [fid.rsplit("_", 1)[0] for fid in feature_ids],
        "mean_high": mean_h, "mean_low": mean_l, "delta": delta, "se": se,
        "tstat": delta / np.maximum(se, 1e-9),
        "pvalue": rng.uniform(0, 0.1, len(feature_ids)),
        "n_high": [n_samples // 2] * len(feature_ids),
        "n_low": [n_samples // 2] * len(feature_ids),
        "missing_fraction": [0.0] * len(feature_ids),
    })

    eigengene = rng.normal(0, 1, n_samples)
    return ExplainResult(
        module_id=module_id,
        gene_driver_table=gene_driver_table,
        transcript_polarity_table=tx_df,
        high_vs_low_table=hvl_table,
        eigengene=eigengene,
        n_module_genes=n_genes,
        sample_ids=sample_ids,
    )


def _make_aligned_feature_table(result: ExplainResult, seed: int = 99) -> pd.DataFrame:
    """DataFrame indexed by sample_id with one column per transcript feature."""
    rng = np.random.default_rng(seed)
    sids = result.sample_ids
    feature_ids = result.transcript_polarity_table["feature_id"].tolist()
    data = {fid: rng.uniform(0, 1, len(sids)) for fid in feature_ids}
    return pd.DataFrame(data, index=sids)


# ---------------------------------------------------------------------------
# plot_driver_bar tests
# ---------------------------------------------------------------------------

def test_plot_driver_bar_returns_axes():
    result = _make_result()
    ax = plot_driver_bar(result)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_driver_bar_respects_top_n():
    result = _make_result(n_genes=30)
    ax = plot_driver_bar(result, top_n=5)
    assert len(ax.patches) == 5


def test_plot_driver_bar_fewer_genes_than_top_n():
    result = _make_result(n_genes=3)
    ax = plot_driver_bar(result, top_n=20)
    assert len(ax.patches) == 3


def test_plot_driver_bar_polarity_coloring():
    result = _make_result(n_genes=20, seed=42)
    ax = plot_driver_bar(result, top_n=20)
    r_vals = result.gene_driver_table.head(20)["r"].to_numpy()
    patches = ax.patches
    for patch, r in zip(patches, r_vals):
        fc = patch.get_facecolor()
        expected = matplotlib.colors.to_rgba("tomato" if r > 0 else "steelblue")
        assert fc == pytest.approx(expected, abs=1e-3), f"r={r:.3f}: expected {'tomato' if r > 0 else 'steelblue'}"


def test_plot_driver_bar_ci_in_unit_range():
    result = _make_result(n_genes=10)
    ax = plot_driver_bar(result, top_n=10)
    for container in ax.containers:
        if hasattr(container, "get_children"):
            continue
    assert ax.get_xlim()[0] >= 0
    assert ax.get_xlim()[1] <= 1


def test_plot_driver_bar_uses_provided_ax():
    import matplotlib.pyplot as plt
    result = _make_result()
    _, provided_ax = plt.subplots()
    returned_ax = plot_driver_bar(result, ax=provided_ax)
    assert returned_ax is provided_ax
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_eigengene_heatmap tests
# ---------------------------------------------------------------------------

def test_plot_eigengene_heatmap_returns_figure_from_dict():
    results = {
        "M000": _make_result(seed=0, module_id="M000"),
        "M001": _make_result(seed=1, module_id="M001"),
    }
    fig = plot_eigengene_heatmap(results)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_eigengene_heatmap_returns_figure_from_single():
    result = _make_result()
    fig = plot_eigengene_heatmap(result)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_eigengene_heatmap_image_shape():
    results = {
        "M000": _make_result(n_genes=5, n_samples=20, seed=0, module_id="M000"),
        "M001": _make_result(n_genes=5, n_samples=20, seed=1, module_id="M001"),
        "M002": _make_result(n_genes=5, n_samples=20, seed=2, module_id="M002"),
    }
    fig = plot_eigengene_heatmap(results)
    ax = fig.axes[0]
    images = ax.get_images()
    assert len(images) == 1
    data = images[0].get_array()
    assert data.shape == (3, 20)


def test_plot_eigengene_heatmap_samples_sorted():
    """Heatmap columns should be sorted so the image data is monotone in sort key."""
    results = {
        "M000": _make_result(n_samples=30, seed=5, module_id="M000"),
        "M001": _make_result(n_samples=30, seed=6, module_id="M001"),
    }
    fig = plot_eigengene_heatmap(results)
    ax = fig.axes[0]
    data = ax.get_images()[0].get_array()
    sort_key = data.mean(axis=0)
    assert np.all(sort_key[:-1] <= sort_key[1:] + 1e-9)


# ---------------------------------------------------------------------------
# plot_high_vs_low_violin tests
# ---------------------------------------------------------------------------

def test_plot_high_vs_low_returns_figure():
    result = _make_result()
    fig = plot_high_vs_low_violin(result)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_high_vs_low_respects_feature_ids():
    result = _make_result(n_genes=10)
    feature_ids = ["G0000_T0", "G0002_T0", "G0004_T0"]
    fig = plot_high_vs_low_violin(result, feature_ids=feature_ids)
    ax = fig.axes[0]
    assert len(ax.get_xticks()) == 3


def test_plot_high_vs_low_default_top_n():
    result = _make_result(n_genes=30)
    fig = plot_high_vs_low_violin(result)
    ax = fig.axes[0]
    assert len(ax.get_xticks()) <= 10


# ---------------------------------------------------------------------------
# plot_transcript_polarity_heatmap tests
# ---------------------------------------------------------------------------

def test_plot_transcript_polarity_heatmap_returns_figure():
    result = _make_result_with_transcripts()
    fig = plot_transcript_polarity_heatmap(result)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_transcript_polarity_heatmap_image_rows():
    result = _make_result_with_transcripts(n_genes=6, n_tx_per_gene=2)
    fig = plot_transcript_polarity_heatmap(result, top_n=4)
    ax = fig.axes[0]
    data = ax.get_images()[0].get_array()
    assert data.shape[0] <= 4


def test_plot_transcript_polarity_heatmap_raises_on_empty():
    result = _make_result()  # no transcript table
    with pytest.raises(ValueError, match="empty"):
        plot_transcript_polarity_heatmap(result)


# ---------------------------------------------------------------------------
# plot_switch_pair tests
# ---------------------------------------------------------------------------

def test_plot_switch_pair_returns_figure():
    result = _make_result_with_transcripts()
    fig = plot_switch_pair(result)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_switch_pair_respects_top_n():
    result = _make_result_with_transcripts(n_genes=10)
    fig = plot_switch_pair(result, top_n=3)
    ax = fig.axes[0]
    assert len(ax.get_yticks()) == 3


def test_plot_switch_pair_raises_on_empty():
    result = _make_result()
    with pytest.raises(ValueError, match="empty"):
        plot_switch_pair(result)


# ---------------------------------------------------------------------------
# plot_isoform_gradient tests
# ---------------------------------------------------------------------------

def test_plot_isoform_gradient_returns_figure():
    result = _make_result_with_transcripts()
    ft = _make_aligned_feature_table(result)
    fig = plot_isoform_gradient(result, ft)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_isoform_gradient_subplots_count():
    result = _make_result_with_transcripts(n_genes=8)
    ft = _make_aligned_feature_table(result)
    fig = plot_isoform_gradient(result, ft, top_n=3)
    assert len(fig.axes) == 3


def test_plot_isoform_gradient_raises_on_empty_table():
    result = _make_result()
    ft = pd.DataFrame(index=result.sample_ids)
    with pytest.raises(ValueError, match="empty"):
        plot_isoform_gradient(result, ft)


# ---------------------------------------------------------------------------
# summarize_module tests
# ---------------------------------------------------------------------------

def test_summarize_module_returns_dict():
    result = _make_result_with_transcripts()
    s = summarize_module(result)
    assert isinstance(s, dict)


def test_summarize_module_keys():
    result = _make_result_with_transcripts()
    s = summarize_module(result)
    expected_keys = {
        "module_id", "n_module_genes", "top_driver_genes",
        "n_positive_transcripts", "n_negative_transcripts",
        "n_switch_genes", "top_switch_gene", "top_switch_pair",
        "top_switch_strength", "mean_abs_driver_r",
    }
    assert expected_keys.issubset(s.keys())


def test_summarize_module_no_transcripts():
    result = _make_result()
    s = summarize_module(result)
    assert s["n_positive_transcripts"] == 0
    assert s["top_switch_gene"] is None


# ---------------------------------------------------------------------------
# plot_summary_panel tests
# ---------------------------------------------------------------------------

def test_plot_summary_panel_returns_figure():
    result = _make_result_with_transcripts()
    ft = _make_aligned_feature_table(result)
    fig = plot_summary_panel(result, feature_table=ft)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_summary_panel_no_transcripts():
    result = _make_result()
    fig = plot_summary_panel(result)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_summary_panel_no_feature_table():
    result = _make_result_with_transcripts()
    fig = plot_summary_panel(result)
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# No-mutation tests
# ---------------------------------------------------------------------------

def test_plot_functions_do_not_mutate_result():
    result = _make_result(n_genes=10)
    original_shape = result.gene_driver_table.shape
    original_r = result.gene_driver_table["r"].to_numpy().copy()

    plot_driver_bar(result)
    plot_eigengene_heatmap(result)
    plot_high_vs_low_violin(result)

    assert result.gene_driver_table.shape == original_shape
    np.testing.assert_array_equal(result.gene_driver_table["r"].to_numpy(), original_r)


# ---------------------------------------------------------------------------
# Integration: plot=True / plot=False with explain_module()
# ---------------------------------------------------------------------------

def _build_explain_inputs(tmp_path: Path):
    """Write minimal artifact files and return (artifact_dir, feature_table, feature_meta)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_genes = 5
    n_samples = 20
    rng = np.random.default_rng(7)

    gene_ids = [f"G{i:04d}" for i in range(n_genes)]
    feature_ids = [f"{g}_T0" for g in gene_ids]
    feature_ids += [f"{g}_T1" for g in gene_ids]

    modules_df = pd.DataFrame({"module_id": ["M000"] * n_genes, "gene_id": gene_ids})

    fs_data = {"gene_id": gene_ids}
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    for sid in sample_ids:
        fs_data[sid] = rng.normal(0, 1, n_genes)
    feature_scores_df = pd.DataFrame(fs_data)

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    pq.write_table(pa.Table.from_pandas(modules_df, preserve_index=False), artifact_dir / "modules.parquet")
    pq.write_table(pa.Table.from_pandas(feature_scores_df, preserve_index=False), artifact_dir / "feature_scores.parquet")

    ft_data = {"sample_id": sample_ids}
    for fid in feature_ids:
        ft_data[fid] = rng.uniform(0, 1, n_samples)
    feature_table = pd.DataFrame(ft_data).set_index("sample_id")

    feature_meta = pd.DataFrame({
        "feature_id": feature_ids,
        "gene_id": [fid.split("_")[0] for fid in feature_ids],
        "feature_type": ["transcript_usage"] * len(feature_ids),
    })

    return artifact_dir, feature_table, feature_meta


def test_explain_module_writes_plots_when_plot_true(tmp_path):
    from isograph.explain.core import explain_module

    artifact_dir, feature_table, feature_meta = _build_explain_inputs(tmp_path)
    output_dir = tmp_path / "explain_out"
    config = ExplainConfig(plot=True)

    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )

    assert (output_dir / "eigengene_heatmap.png").exists()
    assert (output_dir / "M000" / "driver_bar.png").exists()
    assert (output_dir / "M000" / "high_vs_low.png").exists()


def test_explain_module_no_plots_by_default(tmp_path):
    from isograph.explain.core import explain_module

    artifact_dir, feature_table, feature_meta = _build_explain_inputs(tmp_path)
    output_dir = tmp_path / "explain_out_no_plot"

    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
    )

    png_files = list(output_dir.rglob("*.png"))
    assert len(png_files) == 0


def test_manifest_includes_plot_files_when_plot_true(tmp_path):
    from isograph.explain.core import explain_module

    artifact_dir, feature_table, feature_meta = _build_explain_inputs(tmp_path)
    output_dir = tmp_path / "explain_plot_manifest"
    config = ExplainConfig(plot=True)

    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )

    manifest = json.loads((output_dir / "module_explanation_manifest.json").read_text())
    assert "plot_files" in manifest
    assert len(manifest["plot_files"]) > 0


def test_manifest_plot_files_empty_when_plot_false(tmp_path):
    from isograph.explain.core import explain_module

    artifact_dir, feature_table, feature_meta = _build_explain_inputs(tmp_path)
    output_dir = tmp_path / "explain_no_plot_manifest"

    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
    )

    manifest = json.loads((output_dir / "module_explanation_manifest.json").read_text())
    assert manifest["plot_files"] == []


def test_explain_module_writes_pdf_when_format_pdf(tmp_path):
    from isograph.explain.core import explain_module

    artifact_dir, feature_table, feature_meta = _build_explain_inputs(tmp_path)
    output_dir = tmp_path / "explain_pdf"
    config = ExplainConfig(plot=True, output_format="pdf")

    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )

    assert (output_dir / "eigengene_heatmap.pdf").exists()
    assert not (output_dir / "eigengene_heatmap.png").exists()


def test_explain_module_writes_both_formats(tmp_path):
    from isograph.explain.core import explain_module

    artifact_dir, feature_table, feature_meta = _build_explain_inputs(tmp_path)
    output_dir = tmp_path / "explain_both"
    config = ExplainConfig(plot=True, output_format=["png", "pdf"])

    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )

    assert (output_dir / "eigengene_heatmap.png").exists()
    assert (output_dir / "eigengene_heatmap.pdf").exists()


def test_module_summary_json_written(tmp_path):
    from isograph.explain.core import explain_module

    artifact_dir, feature_table, feature_meta = _build_explain_inputs(tmp_path)
    output_dir = tmp_path / "explain_summary"
    config = ExplainConfig(plot=True)

    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )

    summary_path = output_dir / "M000" / "module_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["module_id"] == "M000"
    assert "top_driver_genes" in summary
