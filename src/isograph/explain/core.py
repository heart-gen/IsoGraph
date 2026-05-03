"""Module explanation orchestration (Stage 7A)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from isograph import __version__
from isograph.explain.annotation import (
    _ALL_ANNOTATION_COLUMNS,
    annotate_driver_table,
    annotate_transcript_table,
    load_annotation_table,
)
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
    annotation_table: pd.DataFrame | None = None,
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
        annotation_table: Optional structural annotation table (e.g., output of
            isograph annotate-structure). Must have feature_id or transcript_id column
            plus structural label columns. Merged into gene_driver_table and
            transcript_polarity_table for each module.

    Returns:
        Dict mapping module_id → ExplainResult.
    """
    artifact_dir = Path(artifact_dir)
    config = config or ExplainConfig()

    # Stage 8C: normalize annotation table if provided
    _annotation: pd.DataFrame | None = None
    if annotation_table is not None:
        _annotation = load_annotation_table(annotation_table)
        if "gene_id" not in _annotation.columns:
            id_to_gene = feature_meta.set_index("feature_id")["gene_id"]
            matched = _annotation["feature_id"].map(id_to_gene)
            if matched.notna().any():
                _annotation = _annotation.copy()
                _annotation["gene_id"] = matched
            else:
                warnings.warn(
                    "annotation_table has no gene_id column and no feature_ids matched "
                    "feature_meta; annotate_driver_table will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )

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
        if _annotation is not None:
            gene_driver_table = annotate_driver_table(gene_driver_table, _annotation)
            transcript_polarity_table = annotate_transcript_table(
                transcript_polarity_table, _annotation
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
        _write_outputs(results, Path(output_dir), config, feature_table_aligned, _annotation)

    return results


def _write_outputs(
    results: dict[str, ExplainResult],
    output_dir: Path,
    config: ExplainConfig,
    feature_table: pd.DataFrame | None = None,
    annotation_table: pd.DataFrame | None = None,
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

    _first = next(iter(results.values()), None)
    annotation_columns: list[str] = []
    if annotation_table is not None and _first is not None:
        annotation_columns = [
            c for c in _ALL_ANNOTATION_COLUMNS
            if c in _first.gene_driver_table.columns
            or c in _first.transcript_polarity_table.columns
        ]

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
        "annotation_provided": annotation_table is not None,
        "annotation_columns": annotation_columns,
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
