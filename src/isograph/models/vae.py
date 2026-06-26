"""Stage-4 VAE network model."""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from isograph.features.channels import gene_feature_channels, make_feature_scores
from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.features.reliability import (
    degradation_direction,
    gene_switch_estimability,
    gene_switch_reliability,
)
from isograph.models.base import (
    FitArtifacts,
    NetworkModel,
    compute_module_gene_roles,
    compute_trait_associations,
)
from isograph.models.multiplex import (
    project_feature_similarity_to_gene_graph,
    reconstruction_to_similarity,
    select_alpha_abundance,
    select_alpha_switch,
)
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

        log_every = max(1, cfg.n_epochs // 10)
        t_start = time.time()

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
                    if cfg.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                    optimizer.step()
            else:
                mu, lv = encoder(X_train)
                z = _reparameterize(mu, lv, generator)
                xr = decoder(z)
                loss, _, _ = _elbo_loss(X_train, xr, mu, lv, beta_t)
                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                optimizer.step()

            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                mu_v, lv_v = encoder(X_val)
                xr_v = decoder(mu_v)
                val_loss, recon_l, kl_l = _elbo_loss(X_val, xr_v, mu_v, lv_v, beta_t)

            scheduler.step(val_loss)
            val_f = float(val_loss)

            # Divergence guard: a non-finite val loss means training has blown up
            # (degenerate reconstruction -> downstream near-complete graph / OOM). Stop
            # and fall back to the best checkpoint rather than poison the recon. Always
            # on; only reachable on the pathological path, so healthy runs are unchanged.
            if not math.isfinite(val_f):
                _log.warning(
                    "  latent_dim=%d  epoch %d: non-finite val_loss (%.3g) -- diverged; "
                    "stopping at best_epoch=%d. Set grad_clip_norm (e.g. 1.0) or lower lr.",
                    latent_dim, epoch + 1, val_f, best_epoch,
                )
                early_stopped = True
                break

            if epoch % log_every == 0 or epoch == cfg.n_epochs - 1:
                elapsed = time.time() - t_start
                frac = max((epoch + 1) / cfg.n_epochs, 1e-8)
                eta = elapsed / frac - elapsed
                _log.info(
                    "  latent_dim=%d  epoch %d/%d  val_loss=%.4f  elapsed=%.0fs  eta=%.0fs",
                    latent_dim, epoch + 1, cfg.n_epochs, val_f, elapsed, eta,
                )

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


def _build_node_diagnostics(
    feature_info: pd.DataFrame,
    edge_rows: list[dict],
    module_table: pd.DataFrame,
    gene_reliability: dict[str, float] | None,
    node_stats: dict[str, dict[str, float]],
    alpha_switch: float | None,
    min_module_size: int,
    reliability_on: bool,
) -> pd.DataFrame:
    """Per-gene node-fate diagnostic: explain why a target gene is (or is not) in a module.

    One row per gene. ``fate`` is one of:
      - ``assigned``               -- in a final module (>= min_module_size).
      - ``sub_min_module_size``    -- connected, but its community is below min_module_size.
      - ``downweighted_below_alpha`` -- isolated switch gene whose strongest *raw*
        switch association cleared alpha_switch but reliability downweighting pushed
        its best edge below threshold (only when reliability weighting is on).
      - ``isolated_below_alpha``   -- isolated; no association reached the edge
        threshold (the alpha sparsity floor), reliability not responsible.

    Evidence columns: ``has_switch_feature``/``has_abundance_feature``,
    ``switch_reliability`` (estimability/degradation weight; NaN when weighting off),
    ``max_abs_assoc`` and ``max_switch_assoc`` (strongest raw association to any
    partner; NaN means below the min-alpha pre-filter, i.e. effectively independent),
    and per-channel/total edge ``degree``.
    """
    gene_ids = sorted(feature_info["gene_id"].astype(str).unique())
    types_by_gene = (
        feature_info.assign(gene_id=feature_info["gene_id"].astype(str),
                            feature_type=feature_info["feature_type"].astype(str))
        .groupby("gene_id")["feature_type"].agg(set)
    )

    # Per-gene surviving-edge degree by channel (a gene's endpoint feature_type).
    n_switch_edges: dict[str, int] = {}
    n_abundance_edges: dict[str, int] = {}
    for row in edge_rows:
        for gene_key, type_key in (
            (str(row["source"]), str(row["source_feature_type"])),
            (str(row["target"]), str(row["target_feature_type"])),
        ):
            if type_key == "switch":
                n_switch_edges[gene_key] = n_switch_edges.get(gene_key, 0) + 1
            else:
                n_abundance_edges[gene_key] = n_abundance_edges.get(gene_key, 0) + 1

    assigned = dict(
        zip(module_table["gene_id"].astype(str), module_table["module_id"].astype(str))
    ) if not module_table.empty else {}

    rows = []
    for gene_id in gene_ids:
        gtypes = types_by_gene.get(gene_id, set())
        has_switch = "switch" in gtypes
        has_abundance = "abundance" in gtypes
        rel = (
            float(gene_reliability.get(gene_id, 1.0))
            if (reliability_on and gene_reliability is not None)
            else float("nan")
        )
        stats = node_stats.get(gene_id, {})
        max_abs_assoc = float(stats.get("max_abs_assoc", float("nan")))
        max_switch_assoc = float(stats.get("max_switch_assoc", float("nan")))
        n_sw = n_switch_edges.get(gene_id, 0)
        n_ab = n_abundance_edges.get(gene_id, 0)
        degree = n_sw + n_ab
        module_id = assigned.get(gene_id)

        if module_id is not None:
            fate = "assigned"
        elif degree > 0:
            fate = "sub_min_module_size"
        elif (
            reliability_on
            and has_switch
            and np.isfinite(rel) and rel < 1.0
            and alpha_switch is not None
            and np.isfinite(max_switch_assoc)
            and max_switch_assoc >= alpha_switch
        ):
            fate = "downweighted_below_alpha"
        else:
            fate = "isolated_below_alpha"

        rows.append({
            "gene_id": gene_id,
            "fate": fate,
            "module_id": module_id,
            "has_switch_feature": has_switch,
            "has_abundance_feature": has_abundance,
            "switch_reliability": rel,
            "max_abs_assoc": max_abs_assoc,
            "max_switch_assoc": max_switch_assoc,
            "n_switch_edges": n_sw,
            "n_abundance_edges": n_ab,
            "degree": degree,
        })
    return pd.DataFrame(rows)


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

        Delegates to :func:`reconstruction_to_similarity` (the single source of truth
        shared with post-hoc multi-tier re-projection) so the edge similarity cannot
        drift between the fit and the re-projection pipeline.
        """
        return reconstruction_to_similarity(x_recon)

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
        gene_counts: np.ndarray | None = None,
        gene_table: pd.DataFrame | None = None,
    ) -> FitArtifacts:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the vae backend. Install it with: pip install torch"
            )

        if self.config.residualize_composition:
            # Regress covariates out of the CLR composition before PC1. The switch
            # rows are therefore residualized exactly once (pre-PC1); only the
            # abundance rows (which have no PC1) get the post-hoc residualization.
            design = build_design_matrix(sample_table, self.config.residualize_covariates)
            switch_matrix, feature_info = gene_feature_channels(
                transcript_counts, transcript_table, gene_counts, gene_table,
                switch_design=design,
            )
            if switch_matrix.size:
                is_abundance = (feature_info["feature_type"] == "abundance").to_numpy()
                if is_abundance.any():
                    switch_matrix[is_abundance] = residualize_rows(
                        switch_matrix[is_abundance], design
                    )
        else:
            switch_matrix, feature_info = gene_feature_channels(
                transcript_counts, transcript_table, gene_counts, gene_table
            )
            if switch_matrix.size:
                design = build_design_matrix(sample_table, self.config.residualize_covariates)
                switch_matrix = residualize_rows(switch_matrix, design)

        # Per-gene switch reliability -> switch-switch edge downweighting. Two
        # sources: degradation-alignment (needs a covariate) or covariate-free
        # isoform estimability (minor-isoform usage support).
        gene_reliability: dict[str, float] | None = None
        if self.config.switch_reliability_weighting:
            source = getattr(self.config, "switch_reliability_source", "degradation")
            if source == "estimability":
                gene_reliability = gene_switch_estimability(
                    transcript_counts, transcript_table,
                    min_minor_usage=getattr(
                        self.config, "switch_estimability_min_minor_usage", 0.1),
                    floor=self.config.switch_reliability_floor,
                    power=self.config.switch_reliability_power,
                )
                if gene_reliability:
                    _log.info(
                        "switch reliability: source=estimability, %d genes scored, "
                        "median r=%.3f, frac r<0.5 = %.3f",
                        len(gene_reliability),
                        float(np.median(list(gene_reliability.values()))),
                        float(np.mean([v < 0.5 for v in gene_reliability.values()])),
                    )
            elif self.config.degradation_covariate:
                direction = degradation_direction(sample_table, self.config.degradation_covariate)
                if direction is not None:
                    gene_reliability = gene_switch_reliability(
                        transcript_counts, transcript_table, direction,
                        floor=self.config.switch_reliability_floor,
                        power=self.config.switch_reliability_power,
                    )
                    _log.info(
                        "switch reliability: source=degradation, covariate=%s, %d genes scored, "
                        "median r=%.3f, frac r<0.5 = %.3f",
                        self.config.degradation_covariate, len(gene_reliability),
                        float(np.median(list(gene_reliability.values()))) if gene_reliability else 1.0,
                        float(np.mean([v < 0.5 for v in gene_reliability.values()])) if gene_reliability else 0.0,
                    )

        n_features = switch_matrix.shape[0]
        net_graph = nx.Graph()
        net_graph.add_nodes_from(sorted(feature_info["gene_id"].astype(str).unique()))

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
        node_stats: dict[str, dict[str, float]] = {}
        x_recon_np: np.ndarray | None = None

        if n_features >= 2:
            cfg = self.config
            device = cfg.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
            _log.info("VaeNetworkModel using device=%s", device)
            n_samples_total = switch_matrix.shape[1]
            if cfg.batch_size is None and n_features > 2000:
                effective_bs = min(64, n_samples_total // 4)
                _log.info(
                    "Large-scale input detected (n_features=%d > 2000): auto-setting "
                    "batch_size=%d. Set VaeModelConfig(batch_size=<value>) to override.",
                    n_features, effective_bs,
                )
                cfg = dataclasses.replace(cfg, batch_size=effective_bs)

            X_np = switch_matrix.T.astype(np.float32)  # (n_samples, n_features)
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
                _log.info(
                    "VAE grid sweep: training latent_dim=%d (n_features=%d, n_samples=%d, device=%s) ...",
                    k, n_features, n_samples_total, device,
                )
                _t_k = time.time()
                x_recon_np, cal_partial, enc, dec = _train_single_vae(
                    X_train, X_val, X_all, X_np, n_features, cfg, k, device
                )
                _log.info(
                    "  latent_dim=%d done in %.0fs  rmse=%.4f  best_epoch=%d  early_stopped=%s",
                    k, time.time() - _t_k, cal_partial["reconstruction_rmse"],
                    cal_partial["vae_best_epoch"], cal_partial["vae_early_stopped"],
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
            resolved_alpha_switch = cfg.alpha_switch
            if cfg.alpha_switch_grid is not None:
                resolved_alpha_switch, switch_sweep = select_alpha_switch(
                    partial, feature_info, cfg.alpha_switch_grid,
                )
                calibration["selected_alpha_switch"] = resolved_alpha_switch
                calibration["alpha_switch_sweep"] = switch_sweep
                _log.info(
                    "alpha_switch_grid sweep: selected %.2f from %s",
                    resolved_alpha_switch, cfg.alpha_switch_grid,
                )
                for s in switch_sweep:
                    _log.info(
                        "  alpha_switch=%.2f: n_connected=%d giant_size=%d "
                        "giant_frac=%.3f n_modules_ge30=%d",
                        s["alpha_switch"], s["n_switch_connected"],
                        s["giant_size"], s["giant_fraction"], s["n_modules_ge30"],
                    )
            resolved_alpha_abundance = cfg.alpha_abundance
            if cfg.alpha_abundance_grid is not None:
                resolved_alpha_abundance = select_alpha_abundance(
                    partial, feature_info, cfg.alpha, cfg.alpha_abundance_grid,
                    alpha_switch=resolved_alpha_switch,
                )
                calibration["selected_alpha_abundance"] = resolved_alpha_abundance
            net_graph, edge_rows = project_feature_similarity_to_gene_graph(
                partial, feature_info, cfg.alpha,
                allow_abundance_abundance=cfg.allow_abundance_abundance,
                alpha_switch=resolved_alpha_switch,
                alpha_abundance=resolved_alpha_abundance,
                gene_reliability=gene_reliability,
                node_stats=node_stats,
            )

            if cfg.checkpoint_dir is not None:
                chk_dir = Path(cfg.checkpoint_dir)
                chk_dir.mkdir(parents=True, exist_ok=True)
                chk_path = chk_dir / "vae_checkpoint.pt"
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "n_genes": n_features,
                        "feature_ids": feature_info["feature_id"].astype(str).tolist(),
                        "gene_ids": feature_info["gene_id"].astype(str).tolist(),
                        "feature_types": feature_info["feature_type"].astype(str).tolist(),
                        "latent_dim": selected_k,
                        "hidden_dim": cfg.hidden_dim,
                        "n_hidden_layers": cfg.n_hidden_layers,
                    },
                    chk_path,
                )
                _log.info("Saved VAE checkpoint to %s", chk_path)

        module_table = self._module_table(net_graph)
        feature_scores = make_feature_scores(switch_matrix, feature_info, sample_table)
        # VAE reconstruction of the full multiplex feature matrix (switch + abundance
        # rows), saved so the gene graph can be re-projected post-hoc under different
        # channel rules (switch-only / switch-primary / full-multiplex tiers) without
        # re-fitting. The edge similarity is computed from this reconstruction, not
        # from feature_scores (which is the raw switch matrix). x_recon_np is
        # (n_samples, n_features); transpose to (n_features, n_samples) for storage.
        feature_reconstruction = (
            make_feature_scores(x_recon_np.T, feature_info, sample_table)
            if x_recon_np is not None
            else None
        )
        trait_table, eigengene_table = self._trait_associations(module_table, feature_scores, sample_table)
        module_gene_roles = compute_module_gene_roles(module_table, feature_scores, sample_table)
        node_diagnostics = _build_node_diagnostics(
            feature_info=feature_info,
            edge_rows=edge_rows,
            module_table=module_table,
            gene_reliability=gene_reliability,
            node_stats=node_stats,
            alpha_switch=self.config.alpha_switch,
            min_module_size=self.config.min_module_size,
            reliability_on=self.config.switch_reliability_weighting,
        )

        checkpoint_path: Path | None = None
        if n_features >= 2 and self.config.checkpoint_dir is not None:
            checkpoint_path = Path(self.config.checkpoint_dir) / "vae_checkpoint.pt"

        return FitArtifacts(
            module_table=module_table,
            edge_table=pd.DataFrame(edge_rows),
            trait_table=trait_table,
            feature_scores=feature_scores,
            calibration=calibration,
            checkpoint_path=checkpoint_path,
            eigengene_table=eigengene_table,
            module_gene_roles=module_gene_roles,
            node_diagnostics=node_diagnostics,
            feature_reconstruction=feature_reconstruction,
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
