"""Module explanation orchestration (Stage 7A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from isograph import __version__
from isograph.explain.associations import (
    compute_eigengene,
    compute_gene_driver_table,
    compute_high_vs_low_table,
    compute_transcript_polarity_table,
)
from isograph.explain.config import ExplainConfig, ExplainResult
from isograph.explain.io import load_explain_inputs
from isograph.utils import ensure_dir, write_json


def explain_module(
    artifact_dir: Path | str,
    feature_table: pd.DataFrame,
    feature_meta: pd.DataFrame,
    module_ids: list[str] | None = None,
    output_dir: Path | str | None = None,
    module_score_table: pd.DataFrame | None = None,
    sample_meta: pd.DataFrame | None = None,
    config: ExplainConfig | None = None,
) -> dict[str, ExplainResult]:
    """Explain one or more modules from a fitted IsoGraph artifact.

    Args:
        artifact_dir: Path to directory containing modules.parquet and feature_scores.parquet.
        feature_table: DataFrame with sample_ids as index (or 'sample_id' column) and
            feature_ids as columns.
        feature_meta: DataFrame with columns feature_id, gene_id, feature_type, and
            optional gene_name, transcript_id, exon_id, event_id, source_coordinate.
        module_ids: Module IDs to explain. None = all modules.
        output_dir: If provided, writes per-module parquets and manifest.json here.
        module_score_table: Optional samples × modules DataFrame to override computed eigengenes.
        sample_meta: Reserved for future use (covariates, subsetting).
        config: Statistical configuration. Defaults to ExplainConfig().

    Returns:
        Dict mapping module_id → ExplainResult.
    """
    artifact_dir = Path(artifact_dir)
    config = config or ExplainConfig()

    (
        modules,
        feature_scores,
        feature_table_aligned,
        feature_meta_validated,
        resolved_ids,
        sample_ids,
        module_score_table_aligned,
    ) = load_explain_inputs(artifact_dir, feature_table, feature_meta, module_ids, module_score_table)

    results: dict[str, ExplainResult] = {}
    for module_id in resolved_ids:
        module_genes = modules.loc[modules["module_id"] == module_id, "gene_id"].tolist()

        if module_score_table_aligned is not None:
            eigengene = module_score_table_aligned[module_id].to_numpy(dtype=float)
        else:
            eigengene = compute_eigengene(feature_scores, module_genes, sample_ids)

        gene_driver_table = compute_gene_driver_table(
            feature_scores, eigengene, module_genes, sample_ids, config
        )
        transcript_polarity_table = compute_transcript_polarity_table(
            feature_table_aligned, feature_meta_validated, eigengene, sample_ids, module_genes, config
        )
        high_vs_low_table = compute_high_vs_low_table(
            feature_table_aligned, feature_meta_validated, eigengene, sample_ids, config
        )
        results[module_id] = ExplainResult(
            module_id=module_id,
            gene_driver_table=gene_driver_table,
            transcript_polarity_table=transcript_polarity_table,
            high_vs_low_table=high_vs_low_table,
            eigengene=eigengene,
            n_module_genes=len(module_genes),
            sample_ids=list(sample_ids),
        )

    if output_dir is not None:
        _write_outputs(results, Path(output_dir), config, feature_table_aligned)

    return results


def _write_outputs(
    results: dict[str, ExplainResult],
    output_dir: Path,
    config: ExplainConfig,
    feature_table: pd.DataFrame | None = None,
) -> None:
    ensure_dir(output_dir)
    for module_id, result in results.items():
        module_dir = ensure_dir(output_dir / module_id)
        result.gene_driver_table.to_parquet(module_dir / "gene_driver_table.parquet", index=False)
        result.transcript_polarity_table.to_parquet(module_dir / "transcript_polarity_table.parquet", index=False)
        result.high_vs_low_table.to_parquet(module_dir / "high_vs_low_table.parquet", index=False)

    plot_files: list[str] = []
    if config.plot:
        plot_files = _write_plots(results, output_dir, config, feature_table)

    manifest = {
        "isograph_version": __version__,
        "module_ids": list(results.keys()),
        "n_modules": len(results),
        "tables_per_module": [
            "gene_driver_table.parquet",
            "transcript_polarity_table.parquet",
            "high_vs_low_table.parquet",
        ],
        "plot_files": plot_files,
    }
    write_json(output_dir / "module_explanation_manifest.json", manifest)


def _write_plots(
    results: dict[str, ExplainResult],
    output_dir: Path,
    config: ExplainConfig,
    feature_table: pd.DataFrame | None = None,
) -> list[str]:
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    formats = [config.output_format] if isinstance(config.output_format, str) else list(config.output_format)

    def _save(fig, rel_stem: str) -> list[str]:
        stems: list[str] = []
        for fmt in formats:
            path = output_dir / f"{rel_stem}.{fmt}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            stems.append(f"{rel_stem}.{fmt}")
        plt.close(fig)
        return stems

    written: list[str] = []

    written.extend(_save(plot_eigengene_heatmap(results), "eigengene_heatmap"))

    for module_id, result in results.items():
        module_dir = ensure_dir(output_dir / module_id)

        written.extend(_save(plot_driver_bar(result).figure, f"{module_id}/driver_bar"))
        written.extend(_save(plot_high_vs_low_violin(result), f"{module_id}/high_vs_low"))

        has_tx = not result.transcript_polarity_table.empty
        if has_tx:
            written.extend(_save(
                plot_transcript_polarity_heatmap(result), f"{module_id}/transcript_polarity"
            ))
            written.extend(_save(plot_switch_pair(result), f"{module_id}/switch_pair"))

        if has_tx and feature_table is not None and result.sample_ids:
            try:
                written.extend(_save(
                    plot_isoform_gradient(result, feature_table), f"{module_id}/isoform_gradient"
                ))
            except ValueError:
                pass

        written.extend(_save(
            plot_summary_panel(result, results=results, feature_table=feature_table),
            f"{module_id}/summary_panel",
        ))

        summary = summarize_module(result)
        write_json(module_dir / "module_summary.json", summary)

    return written
