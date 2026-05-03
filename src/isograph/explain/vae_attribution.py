"""VAE decoder attribution (Stage 8D)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from isograph.explain.config import ExplainConfig


def compute_decoder_jacobian(
    checkpoint_path: Path | str,
    eigengene: np.ndarray,
    feature_scores: pd.DataFrame,
    eps: float = 1.0,
) -> pd.DataFrame:
    """Compute finite-difference Jacobian of the VAE decoder w.r.t. the module latent dim.

    Identifies the latent dimension most correlated with `eigengene`, then perturbs it
    ±eps around the mean latent vector to compute the decoded response per gene.

    Args:
        checkpoint_path: Path to ``vae_checkpoint.pt`` saved by VaeNetworkModel.
        eigengene: 1-D array of length n_samples — the module eigengene scores.
        feature_scores: DataFrame with a ``gene_id`` column plus one column per sample
            (same format as the ``feature_scores.parquet`` artifact).
        eps: Perturbation magnitude in latent space. Default 1.0 (≈ 1 SD for a unit
            normal posterior).

    Returns:
        DataFrame with columns: gene_id, decoded_delta, latent_dim_idx, latent_r.
        One row per gene (all genes in checkpoint order).
    """
    import torch

    from isograph.models.vae import _Decoder, _Encoder

    checkpoint_path = Path(checkpoint_path)
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    latent_dim: int = data["latent_dim"]
    hidden_dim: int = data["hidden_dim"]
    n_hidden: int = data["n_hidden_layers"]
    n_genes: int = data["n_genes"]

    encoder = _Encoder(n_genes, hidden_dim, latent_dim, n_hidden)
    encoder.load_state_dict(data["encoder"])
    encoder.eval()

    decoder = _Decoder(latent_dim, hidden_dim, n_genes, n_hidden)
    decoder.load_state_dict(data["decoder"])
    decoder.eval()

    # Build (n_samples × n_genes) input from feature_scores
    scores_df = feature_scores.set_index("gene_id") if "gene_id" in feature_scores.columns else feature_scores
    gene_ids: list[str] = list(scores_df.index)
    sample_ids = list(scores_df.columns)

    # eigengene must align with sample_ids
    if len(eigengene) != len(sample_ids):
        raise ValueError(
            f"eigengene length {len(eigengene)} does not match "
            f"feature_scores n_samples {len(sample_ids)}"
        )

    X = torch.tensor(scores_df.T.values, dtype=torch.float32)  # (n_samples, n_genes)

    with torch.no_grad():
        z_mu, _ = encoder(X)  # (n_samples, latent_dim)

    z_mu_np: np.ndarray = z_mu.numpy()  # (n_samples, latent_dim)

    # Find the latent dim most correlated with the module eigengene
    eg = eigengene - eigengene.mean()
    eg_std = eg.std()
    latent_r = np.zeros(latent_dim)
    if eg_std > 0:
        for j in range(latent_dim):
            col = z_mu_np[:, j]
            col_std = col.std()
            if col_std > 0:
                latent_r[j] = float(np.dot(eg / eg_std, (col - col.mean()) / col_std) / len(eg))

    j_star: int = int(np.argmax(np.abs(latent_r)))

    # Baseline: mean latent vector across samples
    z_bar = torch.tensor(z_mu_np.mean(axis=0), dtype=torch.float32)  # (latent_dim,)

    e_j = torch.zeros(latent_dim)
    e_j[j_star] = 1.0

    with torch.no_grad():
        out_plus = decoder((z_bar + eps * e_j).unsqueeze(0)).squeeze(0).numpy()
        out_minus = decoder((z_bar - eps * e_j).unsqueeze(0)).squeeze(0).numpy()

    decoded_delta: np.ndarray = (out_plus - out_minus) / (2.0 * eps)

    return pd.DataFrame(
        {
            "gene_id": gene_ids,
            "decoded_delta": decoded_delta.astype(np.float64),
            "latent_dim_idx": j_star,
            "latent_r": float(latent_r[j_star]),
        }
    )


def filter_vae_drivers(
    jacobian_df: pd.DataFrame,
    gene_driver_table: pd.DataFrame,
    config: ExplainConfig,
) -> pd.DataFrame:
    """Apply high-confidence filter to VAE decoder attribution results.

    Merges jacobian_df with gene_driver_table (module genes only) and applies three
    criteria: FDR threshold, |decoded_delta| percentile threshold, and sign agreement.

    Returns the merged DataFrame sorted by |decoded_delta| descending, with added
    columns: decoded_delta_percentile, sign_agreement, passes_filter.
    """
    if gene_driver_table.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "decoded_delta",
                "latent_dim_idx",
                "latent_r",
                "decoded_delta_percentile",
                "sign_agreement",
                "passes_filter",
            ]
        )

    merged = gene_driver_table[["gene_id", "r", "qvalue"]].merge(
        jacobian_df[["gene_id", "decoded_delta", "latent_dim_idx", "latent_r"]],
        on="gene_id",
        how="left",
    )

    abs_delta = merged["decoded_delta"].abs()
    if abs_delta.notna().any() and abs_delta.max() > 0:
        merged["decoded_delta_percentile"] = abs_delta.rank(pct=True) * 100.0
    else:
        merged["decoded_delta_percentile"] = 0.0

    # When the selected latent dim is negatively correlated with the eigengene
    # (latent_r < 0), decoded_delta has the opposite sign from r for true module
    # genes. Multiply by sign(latent_r) to get a direction-corrected delta before
    # the sign comparison.
    latent_direction = np.sign(merged["latent_r"]).replace(0, 1.0)
    merged["sign_agreement"] = (
        np.sign(merged["decoded_delta"] * latent_direction) == np.sign(merged["r"])
    )

    passes_fdr = merged["qvalue"] <= config.vae_fdr_threshold
    passes_percentile = merged["decoded_delta_percentile"] >= config.vae_percentile_threshold
    merged["passes_filter"] = passes_fdr & passes_percentile & merged["sign_agreement"]

    return merged.sort_values("decoded_delta", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
