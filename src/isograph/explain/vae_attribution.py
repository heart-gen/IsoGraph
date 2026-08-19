"""VAE decoder attribution (Stage 8D)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from isograph.explain.config import ExplainConfig
from isograph.features.channels import feature_sample_columns

if TYPE_CHECKING:  # torch is an optional runtime dependency, imported lazily inside functions
    import torch


def _select_module_latent_dim(
    encoder: torch.nn.Module,
    X: torch.Tensor,
    eigengene: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return (j_star, latent_r) where j_star is the latent dim most correlated with eigengene."""
    import torch

    with torch.no_grad():
        z_mu, _ = encoder(X)
    z_mu_np: np.ndarray = z_mu.cpu().numpy()
    latent_dim = z_mu_np.shape[1]

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
    return j_star, latent_r


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(checkpoint_path)
    data = torch.load(checkpoint_path, map_location=device, weights_only=True)
    latent_dim: int = data["latent_dim"]
    hidden_dim: int = data["hidden_dim"]
    n_hidden: int = data["n_hidden_layers"]
    n_genes: int = data["n_genes"]

    encoder = _Encoder(n_genes, hidden_dim, latent_dim, n_hidden).to(device)
    encoder.load_state_dict(data["encoder"])
    encoder.eval()

    decoder = _Decoder(latent_dim, hidden_dim, n_genes, n_hidden).to(device)
    decoder.load_state_dict(data["decoder"])
    decoder.eval()

    # Build (n_samples × n_features) input from feature_scores
    sample_ids = feature_sample_columns(feature_scores)
    if "feature_id" in feature_scores.columns:
        scores_df = feature_scores.set_index("feature_id")[sample_ids]
        feature_ids: list[str] = list(scores_df.index)
        gene_ids = feature_scores["gene_id"].astype(str).tolist()
        default_types = pd.Series(["switch"] * len(feature_scores))
        feature_types = feature_scores.get("feature_type", default_types).astype(str).tolist()
    elif "gene_id" in feature_scores.columns:
        scores_df = feature_scores.set_index("gene_id")[sample_ids]
        feature_ids = list(scores_df.index)
        gene_ids = list(scores_df.index)
        feature_types = ["switch"] * len(scores_df)
    else:
        scores_df = feature_scores
        feature_ids = list(scores_df.index)
        gene_ids = list(scores_df.index)
        feature_types = ["switch"] * len(scores_df)

    # eigengene must align with sample_ids
    if len(eigengene) != len(sample_ids):
        raise ValueError(
            f"eigengene length {len(eigengene)} does not match "
            f"feature_scores n_samples {len(sample_ids)}"
        )

    X = torch.tensor(scores_df.T.values, dtype=torch.float32).to(device)  # (n_samples, n_genes)

    j_star, latent_r = _select_module_latent_dim(encoder, X, eigengene)

    # Baseline: mean latent vector across samples
    with torch.no_grad():
        z_mu_np = encoder(X)[0].cpu().numpy()
    z_bar = torch.tensor(z_mu_np.mean(axis=0), dtype=torch.float32, device=device)  # (latent_dim,)

    e_j = torch.zeros(latent_dim, device=device)
    e_j[j_star] = 1.0

    with torch.no_grad():
        out_plus = decoder((z_bar + eps * e_j).unsqueeze(0)).squeeze(0).cpu().numpy()
        out_minus = decoder((z_bar - eps * e_j).unsqueeze(0)).squeeze(0).cpu().numpy()

    decoded_delta: np.ndarray = (out_plus - out_minus) / (2.0 * eps)

    return pd.DataFrame(
        {
            "gene_id": gene_ids,
            "feature_id": feature_ids,
            "feature_type": feature_types,
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

    jacobian_gene = (
        jacobian_df.assign(_abs_delta=jacobian_df["decoded_delta"].abs())
        .sort_values("_abs_delta", ascending=False)
        .drop_duplicates("gene_id", keep="first")
        .drop(columns=["_abs_delta"])
    )
    keep_cols = [
        column
        for column in [
            "gene_id",
            "feature_id",
            "feature_type",
            "decoded_delta",
            "latent_dim_idx",
            "latent_r",
        ]
        if column in jacobian_gene.columns
    ]
    merged = gene_driver_table[["gene_id", "r", "qvalue"]].merge(
        jacobian_gene[keep_cols],
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
    merged["sign_agreement"] = np.sign(merged["decoded_delta"] * latent_direction) == np.sign(
        merged["r"]
    )

    passes_fdr = merged["qvalue"] <= config.vae_fdr_threshold
    passes_percentile = merged["decoded_delta_percentile"] >= config.vae_percentile_threshold
    merged["passes_filter"] = passes_fdr & passes_percentile & merged["sign_agreement"]

    return merged.sort_values("decoded_delta", key=lambda s: s.abs(), ascending=False).reset_index(
        drop=True
    )
