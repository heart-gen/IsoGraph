"""Stage 8A gate tests: module explanation MVP."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.explain import ExplainConfig, ExplainResult, explain_module


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_explain_inputs(
    n_genes: int = 24,
    n_samples: int = 48,
    n_modules: int = 3,
    n_transcripts_per_gene: int = 2,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)

    gene_ids = [f"G{i:04d}" for i in range(n_genes)]
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    genes_per_module = n_genes // n_modules
    module_assignment = np.repeat(np.arange(n_modules), genes_per_module)[:n_genes]

    # Switch coordinates: signal from module latent + noise
    module_latent = rng.normal(size=(n_modules, n_samples))
    switch_coords = np.vstack([
        module_latent[module_assignment[i]] + rng.normal(0, 0.3, n_samples)
        for i in range(n_genes)
    ])
    feature_scores = pd.DataFrame(switch_coords, columns=sample_ids)
    feature_scores.insert(0, "gene_id", gene_ids)

    modules = pd.DataFrame({
        "gene_id": gene_ids,
        "module_id": [f"M{m:03d}" for m in module_assignment],
    })

    # Transcript usages: two transcripts per gene, proportions sum to ~1
    transcript_ids = [f"G{i:04d}_T{t}" for i in range(n_genes) for t in range(n_transcripts_per_gene)]
    # Base usage drawn from Dirichlet; first transcript weakly correlated with module score
    tx_data = np.zeros((n_samples, len(transcript_ids)))
    for i in range(n_genes):
        signal = module_latent[module_assignment[i]] * 0.2
        p = 0.5 + signal * 0.1
        p = np.clip(p, 0.05, 0.95)
        col0 = i * n_transcripts_per_gene
        tx_data[:, col0] = p + rng.normal(0, 0.05, n_samples)
        tx_data[:, col0 + 1] = 1.0 - tx_data[:, col0] + rng.normal(0, 0.05, n_samples)
        tx_data[:, col0 : col0 + n_transcripts_per_gene] = np.clip(
            tx_data[:, col0 : col0 + n_transcripts_per_gene], 0.01, 0.99
        )

    feature_table = pd.DataFrame(tx_data, index=sample_ids, columns=transcript_ids)

    feature_meta = pd.DataFrame([
        {
            "feature_id": f"G{i:04d}_T{t}",
            "gene_id": f"G{i:04d}",
            "transcript_id": f"G{i:04d}_T{t}",
            "feature_type": "transcript_usage",
        }
        for i in range(n_genes)
        for t in range(n_transcripts_per_gene)
    ])

    return modules, feature_scores, feature_table, feature_meta, sample_ids


def _write_artifact(tmp_path: Path, modules: pd.DataFrame, feature_scores: pd.DataFrame) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    modules.to_parquet(artifact_dir / "modules.parquet", index=False)
    feature_scores.to_parquet(artifact_dir / "feature_scores.parquet", index=False)
    return artifact_dir


# ---------------------------------------------------------------------------
# Basic return structure
# ---------------------------------------------------------------------------


def test_explain_module_returns_all_ids(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta, module_ids=None)
    assert set(results.keys()) == {"M000", "M001", "M002"}


def test_explain_module_subset(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M000"])
    assert list(results.keys()) == ["M000"]


def test_explain_result_is_explain_result_type(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M000"])
    assert isinstance(results["M000"], ExplainResult)


# ---------------------------------------------------------------------------
# gene_driver_table schema and values
# ---------------------------------------------------------------------------


def test_gene_driver_table_schema(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M000"])
    df = results["M000"].gene_driver_table
    assert set(df.columns) >= {"gene_id", "r", "pvalue", "qvalue", "n_samples", "missing_fraction"}
    assert {"feature_id", "feature_type"}.issubset(df.columns)


def test_gene_driver_table_only_module_genes(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta)
    for module_id, result in results.items():
        module_genes = set(modules[modules["module_id"] == module_id]["gene_id"])
        assert set(result.gene_driver_table["gene_id"]).issubset(module_genes)


def test_r_values_in_range(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta)
    for result in results.values():
        r = result.gene_driver_table["r"].dropna().to_numpy()
        assert np.all(r >= -1.0 - 1e-9) and np.all(r <= 1.0 + 1e-9)


def test_qvalues_ge_pvalues(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta)
    for result in results.values():
        df = result.gene_driver_table.dropna(subset=["pvalue", "qvalue"])
        assert (df["qvalue"].to_numpy() >= df["pvalue"].to_numpy() - 1e-12).all()


# ---------------------------------------------------------------------------
# transcript_polarity_table schema and values
# ---------------------------------------------------------------------------


def test_transcript_polarity_schema(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M000"])
    df = results["M000"].transcript_polarity_table
    required = {"feature_id", "gene_id", "r", "pvalue", "qvalue", "n_samples", "missing_fraction", "switch_strength"}
    assert required.issubset(set(df.columns))


def test_transcript_polarity_switch_strength_nonneg(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta)
    for result in results.values():
        ss = result.transcript_polarity_table["switch_strength"].dropna()
        assert (ss >= 0.0).all()


def test_switch_strength_is_gene_level(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta)
    for result in results.values():
        df = result.transcript_polarity_table
        for gene_id, grp in df.groupby("gene_id"):
            ss_vals = grp["switch_strength"].dropna().unique()
            assert len(ss_vals) <= 1, f"gene {gene_id} has multiple switch_strength values"


def test_transcript_ids_present_in_feature_meta(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta)
    all_meta_ids = set(feature_meta["feature_id"])
    for result in results.values():
        assert set(result.transcript_polarity_table["feature_id"]).issubset(all_meta_ids)


# ---------------------------------------------------------------------------
# high_vs_low_table schema and values
# ---------------------------------------------------------------------------


def test_high_vs_low_schema(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M000"])
    df = results["M000"].high_vs_low_table
    required = {"feature_id", "gene_id", "mean_high", "mean_low", "delta", "se", "tstat", "pvalue", "n_high", "n_low", "missing_fraction"}
    assert required.issubset(set(df.columns))


def test_high_vs_low_delta_correct(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M000"])
    df = results["M000"].high_vs_low_table.dropna(subset=["delta", "mean_high", "mean_low"])
    expected = df["mean_high"] - df["mean_low"]
    np.testing.assert_allclose(df["delta"].to_numpy(), expected.to_numpy(), atol=1e-9)


def test_high_vs_low_n_total_le_n_samples(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs(n_samples=48)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    results = explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M000"])
    df = results["M000"].high_vs_low_table
    assert ((df["n_high"] + df["n_low"]) <= 48).all()


# ---------------------------------------------------------------------------
# module_score_table override
# ---------------------------------------------------------------------------


def test_module_score_override(tmp_path):
    modules, feature_scores, feature_table, feature_meta, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)

    rng = np.random.default_rng(99)
    mst = pd.DataFrame(
        rng.normal(size=(len(sample_ids), 3)),
        index=sample_ids,
        columns=["M000", "M001", "M002"],
    )
    results = explain_module(artifact_dir, feature_table, feature_meta, module_score_table=mst)
    np.testing.assert_allclose(results["M000"].eigengene, mst["M000"].to_numpy())


# ---------------------------------------------------------------------------
# Output file writing
# ---------------------------------------------------------------------------


def test_output_files_written(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    out_dir = tmp_path / "explain_out"
    explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M000"], output_dir=out_dir)
    assert (out_dir / "module_explanation_manifest.json").exists()
    assert (out_dir / "M000" / "gene_driver_table.parquet").exists()
    assert (out_dir / "M000" / "transcript_polarity_table.parquet").exists()
    assert (out_dir / "M000" / "high_vs_low_table.parquet").exists()


def test_manifest_content(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    out_dir = tmp_path / "explain_out"
    explain_module(artifact_dir, feature_table, feature_meta, output_dir=out_dir)
    manifest = json.loads((out_dir / "module_explanation_manifest.json").read_text())
    assert "module_ids" in manifest
    assert manifest["n_modules"] == 3
    assert "isograph_version" in manifest
    assert set(manifest["module_ids"]) == {"M000", "M001", "M002"}


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_unknown_module_raises(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    with pytest.raises(ValueError, match="Unknown module"):
        explain_module(artifact_dir, feature_table, feature_meta, module_ids=["M999"])


def test_sample_mismatch_raises(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    bad_table = feature_table.rename(index=lambda s: "X_" + s)
    with pytest.raises(ValueError, match="samples overlap"):
        explain_module(artifact_dir, bad_table, feature_meta)


def test_sample_partial_mismatch_warns(tmp_path):
    modules, feature_scores, feature_table, feature_meta, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    # Drop a few samples — should warn, not raise
    partial_table = feature_table.iloc[:-5].copy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        explain_module(artifact_dir, partial_table, feature_meta)
    assert any(issubclass(warning.category, UserWarning) for warning in w)


def test_missing_feature_meta_column_raises(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    bad_meta = feature_meta.drop(columns=["gene_id"])
    with pytest.raises(ValueError, match="gene_id"):
        explain_module(artifact_dir, feature_table, bad_meta)


def test_missing_artifact_file_raises(tmp_path):
    _, _, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        explain_module(empty_dir, feature_table, feature_meta)


def test_duplicate_sample_id_raises(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    dup_table = pd.concat([feature_table, feature_table.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        explain_module(artifact_dir, dup_table, feature_meta)


def test_duplicate_feature_id_raises(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    dup_meta = pd.concat([feature_meta, feature_meta.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        explain_module(artifact_dir, feature_table, dup_meta)


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------


def test_pairwise_missing_handled(tmp_path):
    modules, feature_scores, feature_table, feature_meta, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    # Introduce NaN in first 10 samples for every feature
    ft_with_nan = feature_table.copy().astype(float)
    ft_with_nan.iloc[:10, :] = np.nan
    results = explain_module(artifact_dir, ft_with_nan, feature_meta)
    for result in results.values():
        df = result.transcript_polarity_table.dropna(subset=["n_samples"])
        assert (df["n_samples"].to_numpy() <= len(sample_ids) - 10).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    r1 = explain_module(artifact_dir, feature_table, feature_meta)
    r2 = explain_module(artifact_dir, feature_table, feature_meta)
    pd.testing.assert_frame_equal(r1["M000"].gene_driver_table, r2["M000"].gene_driver_table)
    pd.testing.assert_frame_equal(r1["M000"].transcript_polarity_table, r2["M000"].transcript_polarity_table)


# ---------------------------------------------------------------------------
# Optional annotation columns absent (should warn, not fail)
# ---------------------------------------------------------------------------


def test_optional_meta_columns_absent_warns(tmp_path):
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    minimal_meta = feature_meta[["feature_id", "gene_id", "feature_type"]].copy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = explain_module(artifact_dir, feature_table, minimal_meta)
    assert results  # did not fail
    assert any(issubclass(warning.category, UserWarning) for warning in w)


# ---------------------------------------------------------------------------
# CLI smoke test (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_explain_cli_roundtrip(tmp_path):
    from isograph.workflow.cli import main

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)

    ft_path = tmp_path / "feature_table.parquet"
    fm_path = tmp_path / "feature_meta.parquet"
    feature_table.reset_index(names="sample_id").to_parquet(ft_path, index=False)
    feature_meta.to_parquet(fm_path, index=False)

    out_dir = str(tmp_path / "cli_out")
    main([
        "explain-module",
        "--artifact-dir", str(artifact_dir),
        "--feature-table", str(ft_path),
        "--feature-meta", str(fm_path),
        "--module-ids", "M000",
        "--output-dir", out_dir,
    ])

    assert (Path(out_dir) / "module_explanation_manifest.json").exists()
    assert (Path(out_dir) / "M000" / "gene_driver_table.parquet").exists()
    df = pd.read_parquet(Path(out_dir) / "M000" / "gene_driver_table.parquet")
    assert "r" in df.columns
