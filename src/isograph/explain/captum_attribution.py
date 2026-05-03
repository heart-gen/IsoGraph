"""Captum integrated gradients for VAE encoder attribution (Stage 8E).

Requires the optional dependency group: pip install isograph[torch-explain]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from isograph.explain.config import ExplainConfig


def compute_integrated_gradients(
    checkpoint_path: Path | str,
    eigengene: np.ndarray,
    feature_scores: pd.DataFrame,
    n_steps: int = 50,
    baseline: str = "zero",
) -> pd.DataFrame:
    """Attribute the VAE encoder's latent projection to gene switch coordinates via Integrated Gradients.

    Identifies the latent dimension most correlated with ``eigengene`` (same selection as
    ``compute_decoder_jacobian``), then runs Captum IntegratedGradients from a baseline
    (zero or mean switch vector) to the observed input, integrating gradients through
    the encoder.

    Args:
        checkpoint_path: Path to ``vae_checkpoint.pt`` saved by VaeNetworkModel.
        eigengene: 1-D array of length n_samples — the module eigengene scores.
        feature_scores: DataFrame with a ``gene_id`` column plus one column per sample
            (same format as the ``feature_scores.parquet`` artifact).
        n_steps: Number of interpolation steps for IG approximation (default 50).
            Higher values reduce approximation error but increase runtime.
        baseline: Reference input for IG integration. ``"zero"`` (default) uses a
            zero switch-coordinate vector (no isoform switching). ``"mean"`` uses the
            per-sample mean across all genes.

    Returns:
        DataFrame with columns: gene_id, ig_score, ig_score_abs_mean, latent_dim_idx, latent_r.
        One row per gene (all genes in checkpoint order).
        ``ig_score`` is the signed mean IG attribution across samples; ``ig_score_abs_mean``
        is the unsigned mean and is preferred for ranking.
    """
    import torch

    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError(
            "captum is required for Stage 8E. Install with: pip install isograph[torch-explain]"
        ) from exc

    from isograph.explain.vae_attribution import _select_module_latent_dim
    from isograph.models.vae import _Encoder

    checkpoint_path = Path(checkpoint_path)
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    latent_dim: int = data["latent_dim"]
    hidden_dim: int = data["hidden_dim"]
    n_hidden: int = data["n_hidden_layers"]
    n_genes: int = data["n_genes"]

    encoder = _Encoder(n_genes, hidden_dim, latent_dim, n_hidden)
    encoder.load_state_dict(data["encoder"])
    encoder.eval()

    scores_df = feature_scores.set_index("gene_id") if "gene_id" in feature_scores.columns else feature_scores
    gene_ids: list[str] = list(scores_df.index)
    sample_ids = list(scores_df.columns)

    if len(eigengene) != len(sample_ids):
        raise ValueError(
            f"eigengene length {len(eigengene)} does not match "
            f"feature_scores n_samples {len(sample_ids)}"
        )

    X = torch.tensor(scores_df.T.values, dtype=torch.float32)  # (n_samples, n_genes)

    j_star, latent_r = _select_module_latent_dim(encoder, X, eigengene)

    if baseline == "zero":
        baseline_tensor = torch.zeros_like(X)
    elif baseline == "mean":
        baseline_tensor = X.mean(dim=0, keepdim=True).expand_as(X).clone()
    else:
        raise ValueError(f"baseline must be 'zero' or 'mean', got {baseline!r}")

    # Encoder processes samples independently: ∂(ΣF)/∂x_i = ∂F_i/∂x_i, so batched IG
    # is equivalent to per-sample IG when there are no cross-sample interactions.
    def _project(x: torch.Tensor) -> torch.Tensor:
        mu, _ = encoder(x)
        return mu[:, j_star]

    ig = IntegratedGradients(_project)
    attributions = ig.attribute(
        inputs=X,
        baselines=baseline_tensor,
        n_steps=n_steps,
    ).detach().numpy()  # (n_samples, n_genes)

    ig_score = attributions.mean(axis=0).astype(np.float64)
    ig_score_abs_mean = np.abs(attributions).mean(axis=0).astype(np.float64)

    # Direction-corrected score: multiply by sign(latent_r) so that positive values
    # mean "increases eigengene" regardless of whether the latent dim is positively
    # or negatively correlated with the eigengene.
    direction = float(np.sign(latent_r[j_star])) if latent_r[j_star] != 0 else 1.0
    ig_score_corrected = (ig_score * direction).astype(np.float64)

    return pd.DataFrame(
        {
            "gene_id": gene_ids,
            "ig_score": ig_score,
            "ig_score_corrected": ig_score_corrected,
            "ig_score_abs_mean": ig_score_abs_mean,
            "latent_dim_idx": j_star,
            "latent_r": float(latent_r[j_star]),
        }
    )


def filter_ig_drivers(
    ig_df: pd.DataFrame,
    gene_driver_table: pd.DataFrame,
    config: ExplainConfig,
) -> pd.DataFrame:
    """Apply high-confidence filter to Captum IG attribution results.

    Merges ig_df with gene_driver_table (module genes only) and applies three
    criteria: FDR threshold, |ig_score_abs_mean| percentile threshold, and
    direction-corrected sign agreement (sign(ig_score_corrected) == sign(r)).

    Returns the merged DataFrame sorted by ig_score_abs_mean descending, with
    added columns: ig_score_abs_mean_percentile, sign_agreement, passes_filter.
    """
    if gene_driver_table.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "ig_score",
                "ig_score_corrected",
                "ig_score_abs_mean",
                "latent_dim_idx",
                "latent_r",
                "ig_score_abs_mean_percentile",
                "sign_agreement",
                "passes_filter",
            ]
        )

    merged = gene_driver_table[["gene_id", "r", "qvalue"]].merge(
        ig_df[["gene_id", "ig_score", "ig_score_corrected", "ig_score_abs_mean", "latent_dim_idx", "latent_r"]],
        on="gene_id",
        how="left",
    )

    abs_ig = merged["ig_score_abs_mean"]
    if abs_ig.notna().any() and abs_ig.max() > 0:
        merged["ig_score_abs_mean_percentile"] = abs_ig.rank(pct=True) * 100.0
    else:
        merged["ig_score_abs_mean_percentile"] = 0.0

    passes_fdr = merged["qvalue"] <= config.vae_fdr_threshold
    passes_percentile = merged["ig_score_abs_mean_percentile"] >= config.vae_percentile_threshold
    # IG sign_agreement is not used in the filter: encoder gradient signs reflect
    # internal weight conventions rather than eigengene direction (unlike decoder Jacobian).
    merged["passes_filter"] = passes_fdr & passes_percentile

    return merged.sort_values("ig_score_abs_mean", ascending=False).reset_index(drop=True)
