"""Stage 8D gate tests — VAE decoder attribution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.explain.config import ExplainConfig, ExplainResult
from isograph.explain.vae_attribution import filter_vae_drivers

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_skip_no_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_synthetic_explain_inputs(
    n_genes: int = 12,
    n_samples: int = 30,
    n_modules: int = 2,
    n_transcripts_per_gene: int = 2,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    gene_ids = [f"G{i:04d}" for i in range(n_genes)]
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    genes_per_module = n_genes // n_modules
    module_assignment = np.repeat(np.arange(n_modules), genes_per_module)[:n_genes]
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

    transcript_ids = [f"G{i:04d}_T{t}" for i in range(n_genes) for t in range(n_transcripts_per_gene)]
    tx_data = rng.uniform(0.1, 0.9, (n_samples, len(transcript_ids)))
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


def _make_vae_checkpoint(artifact_dir: Path, n_genes: int, latent_dim: int = 3,
                          hidden_dim: int = 8, n_hidden: int = 1, seed: int = 42) -> Path:
    """Write a randomly initialized VAE checkpoint to artifact_dir/vae_checkpoint.pt."""
    import torch
    from isograph.models.vae import _Decoder, _Encoder

    torch.manual_seed(seed)
    encoder = _Encoder(n_genes, hidden_dim, latent_dim, n_hidden)
    decoder = _Decoder(latent_dim, hidden_dim, n_genes, n_hidden)
    chk_path = artifact_dir / "vae_checkpoint.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "n_genes": n_genes,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "n_hidden_layers": n_hidden,
        },
        chk_path,
    )
    return chk_path


# ---------------------------------------------------------------------------
# Group 1: Config defaults (no torch required)
# ---------------------------------------------------------------------------


def test_explain_config_vae_attribution_default_false():
    cfg = ExplainConfig()
    assert cfg.vae_attribution is False


def test_explain_config_vae_fdr_threshold_default():
    cfg = ExplainConfig()
    assert cfg.vae_fdr_threshold == 0.05


def test_explain_config_vae_percentile_threshold_default():
    cfg = ExplainConfig()
    assert cfg.vae_percentile_threshold == 90.0


def test_explain_config_vae_perturbation_eps_default():
    cfg = ExplainConfig()
    assert cfg.vae_perturbation_eps == 1.0


def test_explain_result_vae_drivers_default_none():
    result = ExplainResult(
        module_id="M000",
        gene_driver_table=pd.DataFrame(),
        transcript_polarity_table=pd.DataFrame(),
        high_vs_low_table=pd.DataFrame(),
        eigengene=np.zeros(10),
        n_module_genes=0,
    )
    assert result.vae_drivers is None


def test_explain_module_vae_drivers_none_when_disabled(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    config = ExplainConfig(vae_attribution=False)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        assert result.vae_drivers is None


def test_explain_module_vae_drivers_none_when_checkpoint_missing(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    # No checkpoint written — should skip gracefully
    config = ExplainConfig(vae_attribution=True)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        assert result.vae_drivers is None


# ---------------------------------------------------------------------------
# Group 2: filter_vae_drivers logic (no torch required)
# ---------------------------------------------------------------------------


def _make_jacobian_df(gene_ids, deltas=None, seed=0):
    rng = np.random.default_rng(seed)
    if deltas is None:
        deltas = rng.normal(0, 1, len(gene_ids))
    return pd.DataFrame({
        "gene_id": gene_ids,
        "decoded_delta": deltas,
        "latent_dim_idx": 0,
        "latent_r": 0.5,
    })


def _make_gene_driver_table(gene_ids, r=None, qvalue=None, seed=1):
    rng = np.random.default_rng(seed)
    n = len(gene_ids)
    if r is None:
        r = rng.uniform(-1, 1, n)
    if qvalue is None:
        qvalue = rng.uniform(0, 1, n)
    return pd.DataFrame({
        "gene_id": gene_ids,
        "r": r,
        "qvalue": qvalue,
        "pvalue": qvalue,
        "n_samples": n,
        "missing_fraction": 0.0,
    })


def test_filter_vae_drivers_empty_gene_driver_table():
    jacobian_df = _make_jacobian_df(["G0", "G1", "G2"])
    gene_driver_table = pd.DataFrame(columns=["gene_id", "pearson_r", "fdr"])
    result = filter_vae_drivers(jacobian_df, gene_driver_table, ExplainConfig())
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_filter_vae_drivers_returns_dataframe_with_required_columns():
    gene_ids = [f"G{i:04d}" for i in range(6)]
    jacobian_df = _make_jacobian_df(gene_ids)
    gene_driver_table = _make_gene_driver_table(gene_ids)
    result = filter_vae_drivers(jacobian_df, gene_driver_table, ExplainConfig())
    for col in ["gene_id", "decoded_delta", "decoded_delta_percentile", "sign_agreement", "passes_filter"]:
        assert col in result.columns, f"Missing column: {col}"


def test_filter_vae_drivers_fdr_filter():
    gene_ids = ["G0", "G1", "G2", "G3"]
    # G0, G1 pass FDR; G2, G3 fail
    jacobian_df = _make_jacobian_df(gene_ids, deltas=[0.8, 0.7, 0.6, 0.5])
    gene_driver_table = _make_gene_driver_table(
        gene_ids,
        r=[0.5, 0.5, 0.5, 0.5],
        qvalue=[0.01, 0.04, 0.06, 0.10],
    )
    config = ExplainConfig(vae_fdr_threshold=0.05, vae_percentile_threshold=0.0)
    result = filter_vae_drivers(jacobian_df, gene_driver_table, config)
    passes = result.set_index("gene_id")["passes_filter"]
    assert passes["G0"]
    assert passes["G1"]
    assert not passes["G2"]
    assert not passes["G3"]


def test_filter_vae_drivers_sign_agreement_filter():
    gene_ids = ["G0", "G1"]
    # G0: same sign (both positive); G1: opposing signs
    jacobian_df = _make_jacobian_df(gene_ids, deltas=[0.5, -0.5])
    gene_driver_table = _make_gene_driver_table(
        gene_ids,
        r=[0.5, 0.5],
        qvalue=[0.01, 0.01],
    )
    config = ExplainConfig(vae_fdr_threshold=1.0, vae_percentile_threshold=0.0)
    result = filter_vae_drivers(jacobian_df, gene_driver_table, config)
    passes = result.set_index("gene_id")["passes_filter"]
    assert passes["G0"]
    assert not passes["G1"]


def test_filter_vae_drivers_sign_agreement_negative_latent_r():
    """When latent_r < 0, decoded_delta has the opposite sign from r — the direction
    correction should account for this so true module genes still pass."""
    gene_ids = ["G0", "G1"]
    # latent_r=-0.5 means the selected dim is anti-correlated with the eigengene.
    # G0: decoded_delta=-0.5, r=0.5 → corrected = -0.5 * -1 = +0.5 → agrees ✓
    # G1: decoded_delta=0.5,  r=0.5 → corrected =  0.5 * -1 = -0.5 → disagrees ✗
    jacobian_df = pd.DataFrame({
        "gene_id": ["G0", "G1"],
        "decoded_delta": [-0.5, 0.5],
        "latent_dim_idx": [0, 0],
        "latent_r": [-0.5, -0.5],
    })
    gene_driver_table = _make_gene_driver_table(
        gene_ids,
        r=[0.5, 0.5],
        qvalue=[0.01, 0.01],
    )
    config = ExplainConfig(vae_fdr_threshold=1.0, vae_percentile_threshold=0.0)
    result = filter_vae_drivers(jacobian_df, gene_driver_table, config)
    passes = result.set_index("gene_id")["passes_filter"]
    assert passes["G0"], "G0 should pass: direction-corrected delta agrees with r"
    assert not passes["G1"], "G1 should fail: direction-corrected delta opposes r"


def test_filter_vae_drivers_percentile_filter():
    gene_ids = [f"G{i}" for i in range(10)]
    # Give genes varying deltas; top 10% should be the largest
    deltas = np.arange(1.0, 11.0)  # G0=1, ..., G9=10
    jacobian_df = _make_jacobian_df(gene_ids, deltas=deltas)
    gene_driver_table = _make_gene_driver_table(
        gene_ids,
        r=np.ones(10) * 0.5,
        qvalue=np.zeros(10),
    )
    config = ExplainConfig(vae_fdr_threshold=1.0, vae_percentile_threshold=90.0)
    result = filter_vae_drivers(jacobian_df, gene_driver_table, config)
    passes = result.set_index("gene_id")["passes_filter"]
    # G9 has the largest delta (10.0) and should be at top percentile
    assert passes["G9"]
    # G0 has the smallest delta and should NOT pass 90th percentile
    assert not passes["G0"]


def test_filter_vae_drivers_passes_filter_is_conjunction():
    gene_ids = ["G0"]
    # All 3 criteria individually satisfied → passes_filter=True
    jacobian_df = _make_jacobian_df(gene_ids, deltas=[1.0])
    gene_driver_table = _make_gene_driver_table(gene_ids, r=[0.5], qvalue=[0.01])
    config = ExplainConfig(vae_fdr_threshold=0.05, vae_percentile_threshold=0.0)
    result = filter_vae_drivers(jacobian_df, gene_driver_table, config)
    assert result.loc[0, "passes_filter"]


def test_filter_vae_drivers_all_pass():
    gene_ids = [f"G{i}" for i in range(5)]
    jacobian_df = _make_jacobian_df(gene_ids, deltas=np.ones(5))
    gene_driver_table = _make_gene_driver_table(
        gene_ids,
        r=np.ones(5) * 0.5,
        qvalue=np.zeros(5),
    )
    config = ExplainConfig(vae_fdr_threshold=1.0, vae_percentile_threshold=0.0)
    result = filter_vae_drivers(jacobian_df, gene_driver_table, config)
    assert result["passes_filter"].all()


def test_filter_vae_drivers_sorted_by_abs_decoded_delta():
    gene_ids = ["G0", "G1", "G2"]
    jacobian_df = _make_jacobian_df(gene_ids, deltas=[-3.0, 1.0, -2.0])
    gene_driver_table = _make_gene_driver_table(gene_ids, r=[-0.5, 0.5, -0.5], qvalue=[0, 0, 0])
    result = filter_vae_drivers(jacobian_df, gene_driver_table, ExplainConfig())
    abs_vals = result["decoded_delta"].abs().values
    assert abs_vals[0] >= abs_vals[1] >= abs_vals[2]


# ---------------------------------------------------------------------------
# Group 3: compute_decoder_jacobian (torch required)
# ---------------------------------------------------------------------------


@_skip_no_torch
def test_compute_decoder_jacobian_returns_dataframe(tmp_path):
    from isograph.explain.vae_attribution import compute_decoder_jacobian

    n_genes = 8
    _, feature_scores, _, _, _ = _make_synthetic_explain_inputs(n_genes=n_genes, n_samples=20)
    eigengene = np.random.default_rng(0).normal(size=20)
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    chk_path = _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    result = compute_decoder_jacobian(chk_path, eigengene, feature_scores)
    assert isinstance(result, pd.DataFrame)


@_skip_no_torch
def test_compute_decoder_jacobian_columns(tmp_path):
    from isograph.explain.vae_attribution import compute_decoder_jacobian

    n_genes = 8
    _, feature_scores, _, _, _ = _make_synthetic_explain_inputs(n_genes=n_genes, n_samples=20)
    eigengene = np.random.default_rng(0).normal(size=20)
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    chk_path = _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    result = compute_decoder_jacobian(chk_path, eigengene, feature_scores)
    for col in ["gene_id", "decoded_delta", "latent_dim_idx", "latent_r"]:
        assert col in result.columns


@_skip_no_torch
def test_compute_decoder_jacobian_row_count(tmp_path):
    from isograph.explain.vae_attribution import compute_decoder_jacobian

    n_genes = 8
    _, feature_scores, _, _, _ = _make_synthetic_explain_inputs(n_genes=n_genes, n_samples=20)
    eigengene = np.random.default_rng(0).normal(size=20)
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    chk_path = _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    result = compute_decoder_jacobian(chk_path, eigengene, feature_scores)
    assert len(result) == n_genes


@_skip_no_torch
def test_compute_decoder_jacobian_gene_ids_preserved(tmp_path):
    from isograph.explain.vae_attribution import compute_decoder_jacobian

    n_genes = 8
    _, feature_scores, _, _, _ = _make_synthetic_explain_inputs(n_genes=n_genes, n_samples=20)
    eigengene = np.random.default_rng(0).normal(size=20)
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    chk_path = _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    result = compute_decoder_jacobian(chk_path, eigengene, feature_scores)
    expected_genes = feature_scores["gene_id"].tolist()
    assert result["gene_id"].tolist() == expected_genes


@_skip_no_torch
def test_compute_decoder_jacobian_decoded_delta_is_float64(tmp_path):
    from isograph.explain.vae_attribution import compute_decoder_jacobian

    n_genes = 8
    _, feature_scores, _, _, _ = _make_synthetic_explain_inputs(n_genes=n_genes, n_samples=20)
    eigengene = np.random.default_rng(0).normal(size=20)
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    chk_path = _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    result = compute_decoder_jacobian(chk_path, eigengene, feature_scores)
    assert result["decoded_delta"].dtype == np.float64


@_skip_no_torch
def test_compute_decoder_jacobian_latent_dim_idx_in_range(tmp_path):
    from isograph.explain.vae_attribution import compute_decoder_jacobian

    n_genes = 8
    latent_dim = 4
    _, feature_scores, _, _, _ = _make_synthetic_explain_inputs(n_genes=n_genes, n_samples=20)
    eigengene = np.random.default_rng(0).normal(size=20)
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    chk_path = _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=latent_dim)
    result = compute_decoder_jacobian(chk_path, eigengene, feature_scores)
    assert int(result["latent_dim_idx"].iloc[0]) in range(latent_dim)


@_skip_no_torch
def test_compute_decoder_jacobian_eps_symmetric(tmp_path):
    """The finite-difference formula is symmetric: eps=+1 and eps=-1 give identical results."""
    from isograph.explain.vae_attribution import compute_decoder_jacobian

    n_genes = 8
    _, feature_scores, _, _, _ = _make_synthetic_explain_inputs(n_genes=n_genes, n_samples=20)
    eigengene = np.random.default_rng(0).normal(size=20)
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    chk_path = _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    result_pos = compute_decoder_jacobian(chk_path, eigengene, feature_scores, eps=1.0)
    result_neg = compute_decoder_jacobian(chk_path, eigengene, feature_scores, eps=-1.0)
    np.testing.assert_allclose(
        result_pos["decoded_delta"].values,
        result_neg["decoded_delta"].values,
        rtol=1e-5,
    )


@_skip_no_torch
def test_compute_decoder_jacobian_latent_dim_1(tmp_path):
    from isograph.explain.vae_attribution import compute_decoder_jacobian

    n_genes = 8
    _, feature_scores, _, _, _ = _make_synthetic_explain_inputs(n_genes=n_genes, n_samples=20)
    eigengene = np.random.default_rng(0).normal(size=20)
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    chk_path = _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=1)
    result = compute_decoder_jacobian(chk_path, eigengene, feature_scores)
    assert int(result["latent_dim_idx"].iloc[0]) == 0
    assert len(result) == n_genes


# ---------------------------------------------------------------------------
# Group 4: Integration with explain_module (torch required)
# ---------------------------------------------------------------------------


@_skip_no_torch
def test_explain_module_vae_drivers_present_when_checkpoint_exists(tmp_path):
    from isograph.explain.core import explain_module

    n_genes = 12
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs(n_genes=n_genes)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=3)

    config = ExplainConfig(vae_attribution=True)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        assert result.vae_drivers is not None
        assert isinstance(result.vae_drivers, pd.DataFrame)


@_skip_no_torch
def test_explain_module_vae_drivers_has_required_columns(tmp_path):
    from isograph.explain.core import explain_module

    n_genes = 12
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs(n_genes=n_genes)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)

    config = ExplainConfig(vae_attribution=True)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        for col in ["gene_id", "decoded_delta", "passes_filter", "sign_agreement"]:
            assert col in result.vae_drivers.columns


@_skip_no_torch
def test_explain_module_vae_drivers_parquet_written(tmp_path):
    from isograph.explain.core import explain_module

    n_genes = 12
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs(n_genes=n_genes)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    output_dir = tmp_path / "out"

    config = ExplainConfig(vae_attribution=True)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )
    for module_id in results:
        parquet_path = output_dir / module_id / "vae_drivers.parquet"
        assert parquet_path.exists(), f"vae_drivers.parquet missing for {module_id}"


@_skip_no_torch
def test_explain_module_vae_drivers_parquet_not_written_when_disabled(tmp_path):
    from isograph.explain.core import explain_module

    n_genes = 12
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs(n_genes=n_genes)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    output_dir = tmp_path / "out"

    config = ExplainConfig(vae_attribution=False)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )
    for module_id in results:
        parquet_path = output_dir / module_id / "vae_drivers.parquet"
        assert not parquet_path.exists()


@_skip_no_torch
def test_explain_module_manifest_vae_attribution_available_true(tmp_path):
    from isograph.explain.core import explain_module

    n_genes = 12
    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs(n_genes=n_genes)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=n_genes, latent_dim=2)
    output_dir = tmp_path / "out"

    config = ExplainConfig(vae_attribution=True)
    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )
    manifest = json.loads((output_dir / "module_explanation_manifest.json").read_text())
    assert manifest["vae_attribution_available"] is True


def test_explain_module_manifest_vae_attribution_available_false(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    output_dir = tmp_path / "out"

    config = ExplainConfig(vae_attribution=False)
    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )
    manifest = json.loads((output_dir / "module_explanation_manifest.json").read_text())
    assert manifest["vae_attribution_available"] is False


# ---------------------------------------------------------------------------
# Group 5: CLI flags (no torch required)
# ---------------------------------------------------------------------------


def test_cli_vae_attribution_flag_accepted():
    from isograph.workflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "explain-module",
        "--artifact-dir", "/tmp/art",
        "--feature-table", "/tmp/ft.parquet",
        "--feature-meta", "/tmp/fm.parquet",
        "--vae-attribution",
    ])
    assert args.vae_attribution is True


def test_cli_vae_fdr_threshold_flag():
    from isograph.workflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "explain-module",
        "--artifact-dir", "/tmp/art",
        "--feature-table", "/tmp/ft.parquet",
        "--feature-meta", "/tmp/fm.parquet",
        "--vae-fdr-threshold", "0.1",
    ])
    assert args.vae_fdr_threshold == 0.1


def test_cli_vae_percentile_threshold_flag():
    from isograph.workflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "explain-module",
        "--artifact-dir", "/tmp/art",
        "--feature-table", "/tmp/ft.parquet",
        "--feature-meta", "/tmp/fm.parquet",
        "--vae-percentile-threshold", "80.0",
    ])
    assert args.vae_percentile_threshold == 80.0


def test_cli_vae_attribution_default_false():
    from isograph.workflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "explain-module",
        "--artifact-dir", "/tmp/art",
        "--feature-table", "/tmp/ft.parquet",
        "--feature-meta", "/tmp/fm.parquet",
    ])
    assert args.vae_attribution is False
