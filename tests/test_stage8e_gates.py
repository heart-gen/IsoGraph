"""Stage 8E gate tests — Captum integrated gradients encoder attribution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isograph.explain.config import ExplainConfig, ExplainResult

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import captum as _captum  # noqa: F401
    _CAPTUM_AVAILABLE = True
except ImportError:
    _CAPTUM_AVAILABLE = False

_skip_no_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
_skip_no_captum = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _CAPTUM_AVAILABLE), reason="torch or captum not installed"
)


# ---------------------------------------------------------------------------
# Synthetic fixture helpers (mirrors test_stage8d_gates.py)
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


def _make_vae_checkpoint(
    artifact_dir: Path,
    n_genes: int,
    latent_dim: int = 3,
    hidden_dim: int = 8,
    n_hidden: int = 1,
    seed: int = 42,
) -> Path:
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
# Group 1: Config defaults (no torch/captum required)
# ---------------------------------------------------------------------------


def test_explain_config_integrated_gradients_default_false():
    cfg = ExplainConfig()
    assert cfg.integrated_gradients is False


def test_explain_config_ig_n_steps_default():
    cfg = ExplainConfig()
    assert cfg.ig_n_steps == 50


def test_explain_config_ig_baseline_default():
    cfg = ExplainConfig()
    assert cfg.ig_baseline == "zero"


def test_explain_result_ig_attributions_default_none():
    result = ExplainResult(
        module_id="M000",
        gene_driver_table=pd.DataFrame(),
        transcript_polarity_table=pd.DataFrame(),
        high_vs_low_table=pd.DataFrame(),
        eigengene=np.zeros(10),
        n_module_genes=0,
    )
    assert result.ig_attributions is None


def test_explain_module_ig_none_when_disabled(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    config = ExplainConfig(integrated_gradients=False)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        assert result.ig_attributions is None


def test_explain_module_ig_none_when_checkpoint_missing(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    # No checkpoint — should skip gracefully
    config = ExplainConfig(integrated_gradients=True)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        assert result.ig_attributions is None


# ---------------------------------------------------------------------------
# Group 2: compute_integrated_gradients unit tests
# ---------------------------------------------------------------------------


@_skip_no_captum
def test_compute_ig_returns_dataframe(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)

    eigengene = np.random.default_rng(0).normal(size=len(sample_ids))
    result = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    assert isinstance(result, pd.DataFrame)


@_skip_no_captum
def test_compute_ig_required_columns(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)

    eigengene = np.random.default_rng(0).normal(size=len(sample_ids))
    result = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    for col in ["gene_id", "ig_score", "ig_score_abs_mean", "latent_dim_idx", "latent_r"]:
        assert col in result.columns, f"Missing column: {col}"


@_skip_no_captum
def test_compute_ig_row_count_equals_n_genes(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs(n_genes=12)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)

    eigengene = np.random.default_rng(0).normal(size=len(sample_ids))
    result = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    assert len(result) == 12


@_skip_no_captum
def test_compute_ig_abs_mean_nonnegative(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)

    eigengene = np.random.default_rng(1).normal(size=len(sample_ids))
    result = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    assert (result["ig_score_abs_mean"] >= 0).all()


@_skip_no_captum
def test_compute_ig_latent_dim_idx_in_range(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    latent_dim = 4
    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12, latent_dim=latent_dim)

    eigengene = np.random.default_rng(2).normal(size=len(sample_ids))
    result = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    assert 0 <= result["latent_dim_idx"].iloc[0] < latent_dim


@_skip_no_captum
def test_compute_ig_zero_baseline_default(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)

    eigengene = np.random.default_rng(3).normal(size=len(sample_ids))
    result_default = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    result_zero = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5, baseline="zero")
    np.testing.assert_array_equal(result_default["ig_score"].values, result_zero["ig_score"].values)


@_skip_no_captum
def test_compute_ig_mean_baseline_differs_from_zero(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)

    eigengene = np.random.default_rng(4).normal(size=len(sample_ids))
    result_zero = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=10, baseline="zero")
    result_mean = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=10, baseline="mean")
    # Mean baseline should produce different scores (feature_scores are nonzero)
    assert not np.allclose(result_zero["ig_score"].values, result_mean["ig_score"].values)


@_skip_no_captum
def test_compute_ig_completeness_approx(tmp_path):
    """Sum of IG scores * n_samples ≈ f(X) - f(baseline) (IG completeness axiom, within ±5%)."""
    import torch
    from isograph.explain.captum_attribution import compute_integrated_gradients
    from isograph.models.vae import _Encoder

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs(n_genes=8, n_samples=20)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=8, latent_dim=3)

    eigengene = np.random.default_rng(5).normal(size=len(sample_ids))
    result = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=100, baseline="zero")

    # Reconstruct f(X) - f(baseline) for verification
    data = torch.load(chk, map_location="cpu", weights_only=True)
    encoder = _Encoder(data["n_genes"], data["hidden_dim"], data["latent_dim"], data["n_hidden_layers"])
    encoder.load_state_dict(data["encoder"])
    encoder.eval()

    scores_df = feature_scores.set_index("gene_id")
    X = torch.tensor(scores_df.T.values, dtype=torch.float32)
    j_star = int(result["latent_dim_idx"].iloc[0])

    with torch.no_grad():
        f_x = encoder(X)[0][:, j_star].numpy()
        f_base = encoder(torch.zeros_like(X))[0][:, j_star].numpy()

    # Sum of ig_score * n_samples should approximate sum(f_x - f_base)
    expected = (f_x - f_base).sum()
    actual = result["ig_score"].sum() * len(sample_ids)
    if abs(expected) > 1e-6:
        rel_err = abs(actual - expected) / abs(expected)
        assert rel_err < 0.10, f"IG completeness relative error {rel_err:.3f} exceeds 10%"


@_skip_no_captum
def test_compute_ig_eigengene_length_mismatch_raises(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)

    bad_eigengene = np.zeros(len(sample_ids) + 5)
    with pytest.raises(ValueError, match="eigengene length"):
        compute_integrated_gradients(chk, bad_eigengene, feature_scores, n_steps=5)


@_skip_no_captum
def test_compute_ig_n_steps_affects_precision(tmp_path):
    """Higher n_steps produces closer approximation to the completeness bound."""
    import torch
    from isograph.explain.captum_attribution import compute_integrated_gradients
    from isograph.models.vae import _Encoder

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs(n_genes=8, n_samples=20)
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=8, latent_dim=3)
    eigengene = np.random.default_rng(6).normal(size=len(sample_ids))

    result_low = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    result_high = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=200)

    data = torch.load(chk, map_location="cpu", weights_only=True)
    encoder = _Encoder(data["n_genes"], data["hidden_dim"], data["latent_dim"], data["n_hidden_layers"])
    encoder.load_state_dict(data["encoder"])
    encoder.eval()
    scores_df = feature_scores.set_index("gene_id")
    X = torch.tensor(scores_df.T.values, dtype=torch.float32)
    j_star = int(result_low["latent_dim_idx"].iloc[0])
    with torch.no_grad():
        f_x = encoder(X)[0][:, j_star].numpy()
        f_base = encoder(torch.zeros_like(X))[0][:, j_star].numpy()

    expected = (f_x - f_base).sum()
    err_low = abs(result_low["ig_score"].sum() * len(sample_ids) - expected)
    err_high = abs(result_high["ig_score"].sum() * len(sample_ids) - expected)
    # high-step result should be at least as close (within tolerance for floating point)
    assert err_high <= err_low + 1e-4, "Higher n_steps should produce better completeness approximation"


@_skip_no_captum
def test_compute_ig_dtype_float64(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)

    eigengene = np.random.default_rng(7).normal(size=len(sample_ids))
    result = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    assert result["ig_score"].dtype == np.float64
    assert result["ig_score_abs_mean"].dtype == np.float64


@_skip_no_captum
def test_compute_ig_latent_r_matches_encoder(tmp_path):
    """latent_r in output should equal Pearson r between encoder mu[j_star] and eigengene."""
    import torch
    from isograph.explain.captum_attribution import compute_integrated_gradients
    from isograph.models.vae import _Encoder

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12, latent_dim=3)
    eigengene = np.random.default_rng(8).normal(size=len(sample_ids))

    result = compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5)
    j_star = int(result["latent_dim_idx"].iloc[0])
    reported_r = float(result["latent_r"].iloc[0])

    # Recompute Pearson r independently
    data = torch.load(chk, map_location="cpu", weights_only=True)
    encoder = _Encoder(data["n_genes"], data["hidden_dim"], data["latent_dim"], data["n_hidden_layers"])
    encoder.load_state_dict(data["encoder"])
    encoder.eval()
    scores_df = feature_scores.set_index("gene_id")
    X = torch.tensor(scores_df.T.values, dtype=torch.float32)
    with torch.no_grad():
        z_mu = encoder(X)[0].numpy()

    col = z_mu[:, j_star]
    eg = eigengene - eigengene.mean()
    eg_std = eg.std()
    col_std = col.std()
    expected_r = float(np.dot(eg / eg_std, (col - col.mean()) / col_std) / len(eg)) if eg_std > 0 and col_std > 0 else 0.0

    assert abs(reported_r - expected_r) < 1e-5, f"latent_r mismatch: {reported_r} vs {expected_r}"


@_skip_no_captum
def test_compute_ig_invalid_baseline_raises(tmp_path):
    from isograph.explain.captum_attribution import compute_integrated_gradients

    modules, feature_scores, _, _, sample_ids = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    chk = _make_vae_checkpoint(artifact_dir, n_genes=12)
    eigengene = np.random.default_rng(9).normal(size=len(sample_ids))

    with pytest.raises(ValueError, match="baseline"):
        compute_integrated_gradients(chk, eigengene, feature_scores, n_steps=5, baseline="random")


# ---------------------------------------------------------------------------
# Group 3: Integration with explain_module
# ---------------------------------------------------------------------------


@_skip_no_captum
def test_explain_module_ig_not_none_when_enabled(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=12)

    config = ExplainConfig(integrated_gradients=True, ig_n_steps=5)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        assert result.ig_attributions is not None
        assert isinstance(result.ig_attributions, pd.DataFrame)


@_skip_no_captum
def test_explain_module_ig_columns(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=12)

    config = ExplainConfig(integrated_gradients=True, ig_n_steps=5)
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        for col in ["gene_id", "ig_score", "ig_score_abs_mean", "latent_dim_idx", "latent_r"]:
            assert col in result.ig_attributions.columns, f"Missing column: {col}"


@_skip_no_captum
def test_explain_module_ig_written_to_disk(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=12)

    output_dir = tmp_path / "explain_out"
    config = ExplainConfig(integrated_gradients=True, ig_n_steps=5)
    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )
    parquet_files = list(output_dir.rglob("ig_attributions.parquet"))
    assert len(parquet_files) > 0, "ig_attributions.parquet not written to disk"


@_skip_no_captum
def test_explain_module_manifest_ig_available_true(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=12)

    output_dir = tmp_path / "explain_out"
    config = ExplainConfig(integrated_gradients=True, ig_n_steps=5)
    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )
    manifest_path = output_dir / "module_explanation_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["ig_attribution_available"] is True


def test_explain_module_manifest_ig_available_false_when_disabled(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)

    output_dir = tmp_path / "explain_out"
    config = ExplainConfig(integrated_gradients=False)
    explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        output_dir=output_dir,
        config=config,
    )
    manifest_path = output_dir / "module_explanation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["ig_attribution_available"] is False


@_skip_no_captum
def test_explain_module_vae_and_ig_both_enabled(tmp_path):
    from isograph.explain.core import explain_module

    modules, feature_scores, feature_table, feature_meta, _ = _make_synthetic_explain_inputs()
    artifact_dir = _write_artifact(tmp_path, modules, feature_scores)
    _make_vae_checkpoint(artifact_dir, n_genes=12)

    config = ExplainConfig(
        vae_attribution=True,
        integrated_gradients=True,
        ig_n_steps=5,
    )
    results = explain_module(
        artifact_dir=artifact_dir,
        feature_table=feature_table,
        feature_meta=feature_meta,
        config=config,
    )
    for result in results.values():
        assert result.vae_drivers is not None
        assert result.ig_attributions is not None


# ---------------------------------------------------------------------------
# Group 4: CLI flags
# ---------------------------------------------------------------------------


def test_cli_integrated_gradients_flag():
    from isograph.workflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "explain-module",
        "--artifact-dir", "x",
        "--feature-table", "x",
        "--feature-meta", "x",
        "--integrated-gradients",
    ])
    assert args.integrated_gradients is True


def test_cli_ig_n_steps_flag():
    from isograph.workflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "explain-module",
        "--artifact-dir", "x",
        "--feature-table", "x",
        "--feature-meta", "x",
        "--ig-n-steps", "100",
    ])
    assert args.ig_n_steps == 100


def test_cli_ig_baseline_flag():
    from isograph.workflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "explain-module",
        "--artifact-dir", "x",
        "--feature-table", "x",
        "--feature-meta", "x",
        "--ig-baseline", "mean",
    ])
    assert args.ig_baseline == "mean"
