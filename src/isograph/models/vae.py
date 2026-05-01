"""Stage-4 VAE network model."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.features.switch import gene_switch_coordinates
from isograph.models.base import FitArtifacts, NetworkModel, compute_trait_associations
from isograph.workflow.config import VaeModelConfig

_log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    def _hidden_dims(hidden_dim: int, n_hidden: int, shrink: bool) -> list[int]:
        base = [hidden_dim // (2 ** i) for i in range(n_hidden)]
        return base if shrink else list(reversed(base))

    class _Encoder(nn.Module):
        def __init__(self, n_genes: int, hidden_dim: int, latent_dim: int, n_hidden: int) -> None:
            super().__init__()
            dims = _hidden_dims(hidden_dim, n_hidden, shrink=True)
            layers: list[nn.Module] = []
            d = n_genes
            for h in dims:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            self.shared = nn.Sequential(*layers)
            self.mu_head = nn.Linear(d, latent_dim)
            self.lv_head = nn.Linear(d, latent_dim)

        def forward(self, x: torch.Tensor):
            h = self.shared(x)
            return self.mu_head(h), self.lv_head(h)

    class _Decoder(nn.Module):
        def __init__(self, latent_dim: int, hidden_dim: int, n_genes: int, n_hidden: int) -> None:
            super().__init__()
            dims = _hidden_dims(hidden_dim, n_hidden, shrink=False)
            layers: list[nn.Module] = []
            d = latent_dim
            for h in dims:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            layers.append(nn.Linear(d, n_genes))
            self.net = nn.Sequential(*layers)

        def forward(self, z: torch.Tensor):
            return self.net(z)

    def _reparameterize(
        mu: torch.Tensor, log_var: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(
            std.shape,
            generator=generator,
            device=std.device,
            dtype=std.dtype,
        )
        return mu + eps * std

    def _elbo_loss(
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = nn.functional.mse_loss(x_recon, x, reduction="mean") * x.shape[1]
        kl = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())
        return recon + beta * kl, recon, kl

    def _detect_collapse(
        mu: torch.Tensor, log_var: torch.Tensor, threshold: float
    ) -> dict:
        per_dim = (0.5 * (mu.pow(2) + log_var.exp() - 1.0 - log_var)).mean(0)
        collapsed = (per_dim < threshold).nonzero(as_tuple=False).squeeze(1).tolist()
        return {
            "vae_per_dim_kl": per_dim.tolist(),
            "vae_n_collapsed_dims": len(collapsed),
            "vae_collapsed_dim_indices": collapsed,
            "vae_mean_kl": float(per_dim.mean()),
            "vae_posterior_collapse": len(collapsed) > 0,
        }

    def _select_latent_dim_idx(
        k_grid: list[int], rmses: list[float], abs_tol: float = 0.01
    ) -> int:
        """Return index of selected k: smallest k where RMSE improvement < abs_tol."""
        for i in range(1, len(k_grid)):
            if rmses[i - 1] - rmses[i] < abs_tol:
                return i - 1
        return len(k_grid) - 1

    def _train_single_vae(
        X_train: torch.Tensor,
        X_val: torch.Tensor,
        X_all: torch.Tensor,
        X_np: np.ndarray,
        n_genes: int,
        cfg,
        latent_dim: int,
        device: str = "cpu",
    ) -> tuple:
        """Train one VAE with a specific latent_dim. Returns (x_recon_np, cal_partial, encoder, decoder)."""
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)

        encoder = _Encoder(n_genes, cfg.hidden_dim, latent_dim, cfg.n_hidden_layers).to(device)
        decoder = _Decoder(latent_dim, cfg.hidden_dim, n_genes, cfg.n_hidden_layers).to(device)

        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.random_state)

        warmup = cfg.warmup_epochs if cfg.warmup_epochs is not None else max(1, cfg.n_epochs // 4)
        best_val = float("inf")
        patience_ctr = 0
        best_epoch = 0
        early_stopped = False
        best_enc_state: dict = {}
        best_dec_state: dict = {}
        best_recon_l = 0.0
        best_kl_l = 0.0

        batch_size = cfg.batch_size
        use_minibatch = batch_size is not None and batch_size < X_train.shape[0]

        for epoch in range(cfg.n_epochs):
            beta_t = min(cfg.beta, cfg.beta * (epoch + 1) / warmup)

            encoder.train()
            decoder.train()

            if use_minibatch:
                perm = torch.randperm(X_train.shape[0], generator=generator, device=device)
                for start in range(0, X_train.shape[0], batch_size):
                    batch = X_train[perm[start : start + batch_size]]
                    mu, lv = encoder(batch)
                    z = _reparameterize(mu, lv, generator)
                    xr = decoder(z)
                    loss, _, _ = _elbo_loss(batch, xr, mu, lv, beta_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                mu, lv = encoder(X_train)
                z = _reparameterize(mu, lv, generator)
                xr = decoder(z)
                loss, _, _ = _elbo_loss(X_train, xr, mu, lv, beta_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                mu_v, lv_v = encoder(X_val)
                xr_v = decoder(mu_v)
                val_loss, recon_l, kl_l = _elbo_loss(X_val, xr_v, mu_v, lv_v, beta_t)

            scheduler.step(val_loss)
            val_f = float(val_loss)

            if epoch >= warmup:
                if val_f < best_val - cfg.early_stop_tol:
                    best_val = val_f
                    best_epoch = epoch
                    patience_ctr = 0
                    best_recon_l = float(recon_l)
                    best_kl_l = float(kl_l)
                    best_enc_state = {k: v.clone() for k, v in encoder.state_dict().items()}
                    best_dec_state = {k: v.clone() for k, v in decoder.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg.patience:
                        early_stopped = True
                        break

        epochs_trained = (best_epoch + 1) if early_stopped else cfg.n_epochs

        if best_enc_state:
            encoder.load_state_dict(best_enc_state)
            decoder.load_state_dict(best_dec_state)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            mu_all, lv_all = encoder(X_all)
            x_recon_all = decoder(mu_all)

        collapse_dict = _detect_collapse(mu_all, lv_all, cfg.collapse_threshold)
        x_recon_np = x_recon_all.detach().cpu().numpy()
        rmse = float(np.sqrt(np.mean((X_np - x_recon_np) ** 2)))

        cal_partial = {
            "reconstruction_rmse": rmse,
            "vae_final_elbo": best_val,
            "vae_final_recon_loss": best_recon_l,
            "vae_final_kl_loss": best_kl_l,
            "vae_best_epoch": best_epoch,
            "vae_early_stopped": early_stopped,
            "vae_n_epochs_trained": epochs_trained,
            "vae_beta_used": cfg.beta,
            "vae_latent_dim": latent_dim,
            "converged": early_stopped,
            **collapse_dict,
        }
        return x_recon_np, cal_partial, encoder, decoder

    _TORCH_AVAILABLE = True

except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


@dataclass
class VaeNetworkModel(NetworkModel):
    config: VaeModelConfig

    def _gene_similarity(self, x_recon: np.ndarray) -> np.ndarray:
        """Pearson correlation on VAE reconstruction.

        x_recon: (n_samples, n_genes) decoded reconstruction.
        Returns (n_genes, n_genes) matrix with zeros on diagonal.
        The VAE's non-linear encoder linearises both linear and non-linear
        module structure into the reconstruction, so Pearson correlation
        captures both types without requiring separate inference modes.
        """
        sim = np.corrcoef(x_recon.T)
        np.fill_diagonal(sim, 0.0)
        return sim

    def _module_table(self, graph: nx.Graph) -> pd.DataFrame:
        rows = []
        for idx, nodes in enumerate(
            sorted(nx.connected_components(graph), key=len, reverse=True)
        ):
            if len(nodes) < self.config.min_module_size:
                continue
            for gene_id in sorted(nodes):
                rows.append({"gene_id": gene_id, "module_id": f"M{idx:03d}"})
        return pd.DataFrame(rows)

    def _trait_associations(
        self,
        module_table: pd.DataFrame,
        feature_scores: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return compute_trait_associations(module_table, feature_scores, sample_table, self.config.trait_columns)

    def fit(
        self,
        transcript_counts: np.ndarray,
        transcript_table: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> FitArtifacts:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the vae backend. Install it with: pip install torch"
            )

        switch_matrix, feature_info = gene_switch_coordinates(transcript_counts, transcript_table)
        if switch_matrix.size:
            design = build_design_matrix(sample_table, self.config.residualize_covariates)
            switch_matrix = residualize_rows(switch_matrix, design)

        n_genes = switch_matrix.shape[0]
        gene_ids = feature_info["gene_id"].tolist()
        net_graph = nx.Graph()
        net_graph.add_nodes_from(gene_ids)

        calibration: dict = {
            "reconstruction_rmse": None,
            "vae_final_elbo": None,
            "vae_final_recon_loss": None,
            "vae_final_kl_loss": None,
            "vae_best_epoch": 0,
            "vae_early_stopped": False,
            "vae_n_epochs_trained": 0,
            "vae_beta_used": self.config.beta,
            "vae_latent_dim": self.config.latent_dim,
            "converged": True,
            "vae_per_dim_kl": [],
            "vae_n_collapsed_dims": 0,
            "vae_collapsed_dim_indices": [],
            "vae_mean_kl": 0.0,
            "vae_posterior_collapse": False,
        }
        edge_rows: list[dict] = []

        if n_genes >= 2:
            cfg = self.config
            device = cfg.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
            _log.info("VaeNetworkModel using device=%s", device)
            n_samples_total = switch_matrix.shape[1]
            if cfg.batch_size is None and n_genes > 2000:
                effective_bs = min(64, n_samples_total // 4)
                _log.info(
                    "Large-scale input detected (n_genes=%d > 2000): auto-setting "
                    "batch_size=%d. Set VaeModelConfig(batch_size=<value>) to override.",
                    n_genes, effective_bs,
                )
                cfg = dataclasses.replace(cfg, batch_size=effective_bs)

            X_np = switch_matrix.T.astype(np.float32)  # (n_samples, n_genes)
            n_samples = X_np.shape[0]

            n_val = max(1, int(n_samples * cfg.val_fraction))
            rng = np.random.default_rng(cfg.random_state)
            idx = rng.permutation(n_samples)
            train_idx, val_idx = idx[n_val:], idx[:n_val]

            X_train = torch.tensor(X_np[train_idx]).to(device)
            X_val = torch.tensor(X_np[val_idx]).to(device)
            X_all = torch.tensor(X_np).to(device)

            k_grid = cfg.latent_dim_grid if cfg.latent_dim_grid else [cfg.latent_dim]
            grid_rmses: list[float] = []
            grid_results: list[tuple] = []

            for k in k_grid:
                x_recon_np, cal_partial, enc, dec = _train_single_vae(
                    X_train, X_val, X_all, X_np, n_genes, cfg, k, device
                )
                grid_rmses.append(cal_partial["reconstruction_rmse"])
                grid_results.append((x_recon_np, cal_partial, enc, dec))

            sel_idx = _select_latent_dim_idx(k_grid, grid_rmses) if len(k_grid) > 1 else 0
            selected_k = k_grid[sel_idx]
            x_recon_np, cal_partial, encoder, decoder = grid_results[sel_idx]

            if len(k_grid) > 1:
                cal_partial["latent_dim_selected"] = selected_k
                cal_partial["latent_dim_grid_rmses"] = dict(zip(k_grid, grid_rmses))
                _log.info(
                    "latent_dim_grid sweep: selected k=%d from %s (rmses=%s)",
                    selected_k, k_grid,
                    [f"{r:.4f}" for r in grid_rmses],
                )

            if cal_partial["vae_n_collapsed_dims"] >= max(1, selected_k // 2):
                _log.warning(
                    "VAE posterior collapse: %d/%d latent dims collapsed. "
                    "Consider reducing beta or increasing warmup_epochs.",
                    cal_partial["vae_n_collapsed_dims"],
                    selected_k,
                )

            calibration.update(cal_partial)

            partial = self._gene_similarity(x_recon_np)
            for i, source in enumerate(gene_ids):
                for j in range(i + 1, n_genes):
                    weight = float(partial[i, j])
                    if abs(weight) < cfg.alpha:
                        continue
                    target = gene_ids[j]
                    net_graph.add_edge(source, target, weight=weight)
                    edge_rows.append({"source": source, "target": target, "weight": weight})

            if cfg.checkpoint_dir is not None:
                chk_dir = Path(cfg.checkpoint_dir)
                chk_dir.mkdir(parents=True, exist_ok=True)
                chk_path = chk_dir / "vae_checkpoint.pt"
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "n_genes": n_genes,
                        "latent_dim": selected_k,
                        "hidden_dim": cfg.hidden_dim,
                        "n_hidden_layers": cfg.n_hidden_layers,
                    },
                    chk_path,
                )
                _log.info("Saved VAE checkpoint to %s", chk_path)

        module_table = self._module_table(net_graph)
        feature_scores = pd.DataFrame(switch_matrix, index=feature_info["gene_id"])
        feature_scores = feature_scores.reset_index().rename(columns={"index": "gene_id"})
        trait_table, eigengene_table = self._trait_associations(module_table, feature_scores, sample_table)

        checkpoint_path: Path | None = None
        if n_genes >= 2 and self.config.checkpoint_dir is not None:
            checkpoint_path = Path(self.config.checkpoint_dir) / "vae_checkpoint.pt"

        return FitArtifacts(
            module_table=module_table,
            edge_table=pd.DataFrame(edge_rows),
            trait_table=trait_table,
            feature_scores=feature_scores,
            calibration=calibration,
            checkpoint_path=checkpoint_path,
            eigengene_table=eigengene_table,
        )


def load_vae_checkpoint(
    checkpoint_path: Path, config: VaeModelConfig, n_genes: int
) -> VaeNetworkModel:
    """Reload a VAE model from a saved checkpoint for inference."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required to load VAE checkpoints. Install it with: pip install torch"
        )
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    chk_n_genes = data["n_genes"]
    if chk_n_genes != n_genes:
        raise ValueError(
            f"Checkpoint n_genes={chk_n_genes} does not match requested n_genes={n_genes}"
        )
    latent_dim = data["latent_dim"]
    hidden_dim = data["hidden_dim"]
    n_hidden = data["n_hidden_layers"]

    encoder = _Encoder(n_genes, hidden_dim, latent_dim, n_hidden)
    decoder = _Decoder(latent_dim, hidden_dim, n_genes, n_hidden)
    encoder.load_state_dict(data["encoder"])
    decoder.load_state_dict(data["decoder"])
    encoder.eval()
    decoder.eval()

    model = VaeNetworkModel(config)
    model._loaded_encoder = encoder  # type: ignore[attr-defined]
    model._loaded_decoder = decoder  # type: ignore[attr-defined]
    return model
