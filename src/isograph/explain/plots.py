"""Module explanation plots (Stage 8B)."""

from __future__ import annotations

import numpy as np

# ── existing plots (modified defaults / behaviour) ─────────────────────────


def plot_driver_bar(result, top_n: int = 10, ax=None):
    """Horizontal bar chart of top driver genes by |r| with 95 % CI. Returns Axes."""
    import matplotlib.pyplot as plt

    df = result.gene_driver_table.head(top_n)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(df))))

    r = df["r"].to_numpy(dtype=float)
    colors = np.where(r > 0, "tomato", "steelblue")
    n = df["n_samples"].to_numpy(dtype=float)

    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se_z = 1.0 / np.sqrt(np.maximum(n - 3, 1))
    ci_half = (np.tanh(z + 1.96 * se_z) - np.tanh(z - 1.96 * se_z)) / 2
    xerr = np.clip(ci_half, 0, None)

    y = np.arange(len(df))
    ax.barh(y, np.abs(r), xerr=xerr, color=colors, align="center")
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene_id"])
    ax.invert_yaxis()
    ax.set_xlabel("|r| (Pearson correlation with eigengene)")
    ax.set_xlim(0, 1)

    return ax


def plot_eigengene_heatmap(results, sample_meta=None, ax=None):
    """Heatmap of eigengenes × samples, samples sorted by eigengene score. Returns Figure.

    Args:
        results: dict[str, ExplainResult] or a single ExplainResult.
        sample_meta: Reserved for future use (covariate annotation bars).
        ax: Optional pre-created Axes.
    """
    import matplotlib.pyplot as plt

    if isinstance(results, dict):
        module_ids = list(results.keys())
        mat = np.stack([results[mid].eigengene for mid in module_ids], axis=0)
    else:
        module_ids = [results.module_id]
        mat = results.eigengene[np.newaxis, :]

    n_modules, n_samples = mat.shape

    sort_key = mat.mean(axis=0) if n_modules > 1 else mat[0]
    mat = mat[:, np.argsort(sort_key)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_samples * 0.15), max(2, n_modules * 0.5)))
    else:
        fig = ax.figure

    vmax = float(np.nanmax(np.abs(mat))) or 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(n_modules))
    ax.set_yticklabels(module_ids)
    ax.set_xticks([])
    ax.set_xlabel("Samples (sorted by eigengene)")
    fig.colorbar(im, ax=ax, label="Eigengene score")

    return fig


def plot_high_vs_low_violin(result, feature_ids=None, top_n: int = 10, ax=None):
    """Dumbbell plot: mean feature value in high vs low module-score samples. Returns Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = result.high_vs_low_table

    if feature_ids is not None:
        present = set(df["feature_id"])
        ordered = [fid for fid in feature_ids if fid in present]
        df = df.set_index("feature_id").reindex(ordered).reset_index()
    else:
        df = df.reindex(df["tstat"].abs().nlargest(top_n).index).reset_index(drop=True)

    n = len(df)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, n * 0.5 + 1), 4))
    else:
        fig = ax.figure

    x = np.arange(n)
    line_colors = np.where(
        df["delta"].abs() <= df["se"],
        "gray",
        np.where(df["delta"] > 0, "tomato", "steelblue"),
    )
    ax.vlines(
        x,
        df["mean_low"].to_numpy(),
        df["mean_high"].to_numpy(),
        colors=line_colors,
        lw=1.5,
        zorder=1,
    )
    ax.scatter(x, df["mean_high"].to_numpy(), color="tomato", zorder=2, s=50)
    ax.scatter(
        x,
        df["mean_low"].to_numpy(),
        color="steelblue",
        zorder=2,
        s=50,
        marker="o",
        facecolors="none",
        edgecolors="steelblue",
        linewidths=1.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["feature_id"], rotation=45, ha="right")
    ax.set_ylabel("Mean feature value")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tomato",
                markersize=8,
                label="High module",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="steelblue",
                markersize=8,
                label="Low module",
            ),
        ]
    )

    return fig


# ── new plots ──────────────────────────────────────────────────────────────


def plot_isoform_gradient(result, feature_table, top_n: int = 5, axes=None):
    """Scatter + smoothed-trend of transcript usage vs module eigengene score.

    Args:
        result: ExplainResult with transcript_polarity_table and sample_ids.
        feature_table: DataFrame indexed by sample_id, columns = feature_ids.
        top_n: Number of top driver genes (one subplot each).
        axes: Optional list of pre-created Axes for embedding in a summary panel.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    tx_table = result.transcript_polarity_table
    if tx_table.empty:
        raise ValueError("transcript_polarity_table is empty; no transcript usage features.")

    genes_with_tx = set(tx_table["gene_id"])
    top_genes = [g for g in result.gene_driver_table["gene_id"] if g in genes_with_tx][:top_n]
    if not top_genes:
        raise ValueError("No top driver genes have transcript usage data.")

    if axes is not None:
        top_genes = top_genes[: len(axes)]
    n_genes = len(top_genes)

    if axes is None:
        fig, ax_arr = plt.subplots(1, n_genes, figsize=(4 * n_genes, 3.5), squeeze=False)
        axes_list = list(ax_arr[0])
    else:
        axes_list = list(axes)[:n_genes]
        fig = axes_list[0].figure

    eigengene = result.eigengene
    order = np.argsort(eigengene)
    x_sorted = eigengene[order]
    window = max(3, len(eigengene) // 8)

    norm = plt.Normalize(-1, 1)
    cmap = plt.cm.RdBu_r

    for ax, gene_id in zip(axes_list, top_genes, strict=False):
        gene_txs = tx_table[tx_table["gene_id"] == gene_id].sort_values("r")
        for _, tx_row in gene_txs.iterrows():
            fid = tx_row["feature_id"]
            if fid not in feature_table.columns:
                continue
            usage = (
                feature_table.reindex(result.sample_ids)[fid].to_numpy(dtype=float)
                if result.sample_ids
                else feature_table[fid].to_numpy(dtype=float)
            )
            y_sorted = usage[order]
            color = cmap(norm(float(tx_row["r"])))
            ax.scatter(x_sorted, y_sorted, color=color, s=8, alpha=0.5, zorder=1)
            if np.isfinite(y_sorted).sum() >= window:
                trend = uniform_filter1d(y_sorted, size=window, mode="nearest")
                ax.plot(x_sorted, trend, color=color, lw=2, label=fid, zorder=2)
        ax.set_title(gene_id, fontsize=10)
        ax.set_xlabel("Module score")
        ax.set_ylabel("Transcript usage")
        ax.legend(fontsize=6, loc="best")

    return fig


def plot_transcript_polarity_heatmap(result, top_n: int = 15, ax=None):
    """Heatmap: rows = top driver genes, columns = transcripts, color = signed r. Returns Figure."""
    import matplotlib.pyplot as plt

    tx_table = result.transcript_polarity_table
    if tx_table.empty:
        raise ValueError("transcript_polarity_table is empty; no transcript usage features.")

    top_genes = result.gene_driver_table.head(top_n)["gene_id"].tolist()
    subset = tx_table[tx_table["gene_id"].isin(top_genes)]
    pivot = subset.pivot_table(index="gene_id", columns="feature_id", values="r", aggfunc="first")
    gene_order = [g for g in top_genes if g in pivot.index]
    pivot = pivot.loc[gene_order]
    # Reorder columns to match row (driver-rank) order so blocks are block-diagonal
    col_gene = subset.set_index("feature_id")["gene_id"].to_dict()
    col_order = sorted(
        pivot.columns,
        key=lambda c: gene_order.index(col_gene.get(c, c))
        if col_gene.get(c, c) in gene_order
        else len(gene_order),
    )
    pivot = pivot[col_order]

    mat = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    n_rows, n_cols = mat.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, n_cols * 0.7 + 1), max(2, n_rows * 0.5 + 1)))
    else:
        fig = ax.figure

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("white")
    vmax = float(np.nanmax(np.abs(mat.data))) if mat.count() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(gene_order)

    _LABEL_THRESHOLD = 30
    col_to_gene = subset.set_index("feature_id")["gene_id"].to_dict()
    col_genes = [col_to_gene.get(c, c) for c in pivot.columns]

    if n_cols > _LABEL_THRESHOLD:
        # Group columns by gene; show one tick per gene at midpoint + separator lines
        gene_col_idxs: dict[str, list[int]] = {}
        for idx, g in enumerate(col_genes):
            gene_col_idxs.setdefault(g, []).append(idx)
        mid_ticks = [float(np.mean(idxs)) for idxs in gene_col_idxs.values()]
        ax.set_xticks(mid_ticks)
        ax.set_xticklabels(list(gene_col_idxs.keys()), rotation=45, ha="right", fontsize=8)
        prev = col_genes[0]
        for idx, g in enumerate(col_genes):
            if g != prev:
                ax.axvline(idx - 0.5, color="white", lw=1.5, zorder=2)
                prev = g
    else:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Gene (by driver rank)")
    ax.set_xlabel("Transcript / feature")
    fig.colorbar(im, ax=ax, label="r (association with eigengene)", fraction=0.02, pad=0.04)

    return fig


def plot_switch_pair(result, top_n: int = 5, ax=None):
    """Diverging bar: left = most-negative transcript, right = most-positive, per top switch gene.

    Returns matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    tx_table = result.transcript_polarity_table
    if tx_table.empty:
        raise ValueError("transcript_polarity_table is empty; no transcript usage features.")

    gene_ss = tx_table.groupby("gene_id")["switch_strength"].first()
    top_genes = gene_ss.nlargest(top_n).index.tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(2, len(top_genes) * 0.7 + 0.8)))
    else:
        fig = ax.figure

    for i, gene_id in enumerate(top_genes):
        gene_txs = tx_table[tx_table["gene_id"] == gene_id].sort_values("r")
        if len(gene_txs) < 2:
            continue
        neg_tx = gene_txs.iloc[0]
        pos_tx = gene_txs.iloc[-1]

        ax.barh(i, neg_tx["r"], color="steelblue", height=0.5, align="center")
        ax.text(
            min(float(neg_tx["r"]) - 0.03, -0.05),
            i,
            f"↓ {neg_tx['feature_id']}",
            ha="right",
            va="center",
            fontsize=8,
            color="steelblue",
        )
        ax.barh(i, pos_tx["r"], color="tomato", height=0.5, align="center")
        ax.text(
            max(float(pos_tx["r"]) + 0.03, 0.05),
            i,
            f"↑ {pos_tx['feature_id']}",
            ha="left",
            va="center",
            fontsize=8,
            color="tomato",
        )

    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.8, zorder=0)
    ax.set_xlabel("r (Pearson correlation with module eigengene)")
    ax.set_xlim(-1.5, 1.5)

    return fig


def summarize_module(result) -> dict:
    """Return a structured summary dict for a single ExplainResult."""
    gdt = result.gene_driver_table
    tpt = result.transcript_polarity_table

    top_genes = gdt.head(5)["gene_id"].tolist()

    if not tpt.empty:
        n_pos = int((tpt["r"] > 0).sum())
        n_neg = int((tpt["r"] < 0).sum())
        gene_ss = tpt.groupby("gene_id")["switch_strength"].first()
        n_switch = int((gene_ss > gene_ss.median()).sum())
        top_switch_gene = str(gene_ss.idxmax())
        top_txs = tpt[tpt["gene_id"] == top_switch_gene].sort_values("r")
        top_switch_pair = (
            [top_txs.iloc[0]["feature_id"], top_txs.iloc[-1]["feature_id"]]
            if len(top_txs) >= 2
            else []
        )
        top_switch_strength = float(gene_ss.max())
    else:
        n_pos = n_neg = n_switch = 0
        top_switch_gene = None
        top_switch_pair = []
        top_switch_strength = None

    return {
        "module_id": result.module_id,
        "n_module_genes": result.n_module_genes,
        "top_driver_genes": top_genes,
        "n_positive_transcripts": n_pos,
        "n_negative_transcripts": n_neg,
        "n_switch_genes": n_switch,
        "top_switch_gene": top_switch_gene,
        "top_switch_pair": top_switch_pair,
        "top_switch_strength": top_switch_strength,
        "mean_abs_driver_r": float(gdt["r"].abs().mean()) if not gdt.empty else None,
    }


def plot_summary_panel(
    result,
    results=None,
    feature_table=None,
    top_n_drivers: int = 5,
    top_n_gradient: int = 2,
):
    """5-panel summary figure for a single module.

    Panels: A eigengene heatmap · B driver bar · C transcript polarity heatmap ·
            D isoform gradient · E switch-pair (or dumbbell when no transcripts).

    Args:
        result: ExplainResult for the module being summarised.
        results: Optional dict of all ExplainResults for the multi-module eigengene heatmap in A.
        feature_table: Optional aligned DataFrame (samples × features) for panel D gradient.
        top_n_drivers: Genes shown in B, C, E.
        top_n_gradient: Genes shown as gradient subplots in D.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    has_tx = not result.transcript_polarity_table.empty
    has_ft = feature_table is not None and bool(result.sample_ids)
    show_gradient = has_tx and has_ft

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[0.8, 2.5, 1.8, 1.5],
        width_ratios=[1, 1.5],
        hspace=0.55,
        wspace=0.35,
    )

    def _panel_label(ax, letter, x=-0.02, y=1.05):
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    # A — eigengene heatmap
    ax_a = fig.add_subplot(gs[0, :])
    plot_eigengene_heatmap(results if results is not None else result, ax=ax_a)
    _panel_label(ax_a, "A")

    # B — driver bar
    ax_b = fig.add_subplot(gs[1, 0])
    plot_driver_bar(result, top_n=top_n_drivers, ax=ax_b)
    _panel_label(ax_b, "B", x=-0.15)

    # C — transcript polarity heatmap
    ax_c = fig.add_subplot(gs[1, 1])
    if has_tx:
        plot_transcript_polarity_heatmap(result, top_n=top_n_drivers, ax=ax_c)
    else:
        ax_c.text(
            0.5,
            0.5,
            "No transcript usage features",
            ha="center",
            va="center",
            transform=ax_c.transAxes,
            color="gray",
        )
        ax_c.set_axis_off()
    _panel_label(ax_c, "C", x=-0.08)

    # D — isoform gradient (nested sub-gridspec)
    if show_gradient:
        n_grad = min(top_n_gradient, len(result.gene_driver_table))
        gs_d = gs[2, :].subgridspec(1, n_grad, wspace=0.4)
        axes_d = [fig.add_subplot(gs_d[0, i]) for i in range(n_grad)]
        try:
            plot_isoform_gradient(result, feature_table, top_n=n_grad, axes=axes_d)
        except ValueError:
            for axi in axes_d:
                axi.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axi.transAxes,
                    color="gray",
                )
                axi.set_axis_off()
        _panel_label(axes_d[0], "D", x=-0.15, y=1.1)
    else:
        ax_d = fig.add_subplot(gs[2, :])
        msg = (
            "Pass feature_table for isoform gradient"
            if not has_ft
            else "No transcript usage features"
        )
        ax_d.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=ax_d.transAxes,
            color="gray",
            fontstyle="italic",
        )
        ax_d.set_axis_off()
        _panel_label(ax_d, "D")

    # E — switch pair or dumbbell
    ax_e = fig.add_subplot(gs[3, :])
    if has_tx:
        plot_switch_pair(result, top_n=top_n_drivers, ax=ax_e)
    else:
        plot_high_vs_low_violin(result, ax=ax_e)
    _panel_label(ax_e, "E")

    fig.suptitle(f"Module {result.module_id}", fontsize=14, y=0.99)
    return fig
