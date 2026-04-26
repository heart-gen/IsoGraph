"""Stage-7 GPU-accelerated Factor Analysis network model.

Uses the Woodbury matrix identity to compute FA marginal log-likelihood
without ever materialising the p×p covariance matrix Σ = WW^T + Ψ.
All core operations are O(n·p·k + k³) where k << p.

BIC-based component selection replaces the 5-fold CV used in LatentNetworkModel,
cutting wall-clock selection time from O(|grid| × 5 × max_iter) to O(|grid| × max_iter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

from isograph.features.residualize import build_design_matrix, residualize_rows
from isograph.features.switch import gene_switch_coordinates
from isograph.models.base import FitArtifacts, NetworkModel
from isograph.workflow.config import GpuLatentModelConfig

_log = logging.getLogger(__name__)

try:
    import torch
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Woodbury FA core
# ---------------------------------------------------------------------------

def _fit_fa_woodbury(
    Xc: np.ndarray,   # (n_samples, n_genes), already mean-centred
    k: int,
    max_iter: int,
    lr: float,
    tol: float,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Fit FA loadings W (p×k) and log-noise log_ψ (p,) via gradient descent.

    Returns (W, log_psi, mean_log_ll_per_sample, n_iter).
    The FA marginal log-likelihood is computed via:
      log p(X) = Σ_i -½ [x_i^T Σ^{-1} x_i + log|Σ| + p log(2π)]
    where Σ^{-1} and log|Σ| are evaluated with Woodbury / matrix-determinant lemma,
    never forming the p×p Σ matrix.
    """
    import torch

    torch.manual_seed(seed)
    n, p = Xc.shape
    Xt = torch.tensor(Xc, dtype=torch.float32, device=device)

    W = torch.nn.Parameter(torch.randn(p, k, device=device) * (k ** -0.5))
    log_psi = torch.nn.Parameter(torch.zeros(p, device=device))

    opt = optim.Adam([W, log_psi], lr=lr)
    log_2pi = float(np.log(2.0 * np.pi))

    prev_ll = -float("inf")
    n_iter = max_iter
    mean_nll = torch.tensor(float("inf"))

    for step in range(max_iter):
        opt.zero_grad()

        psi_inv = torch.exp(-log_psi)                          # (p,)
        M = torch.eye(k, device=device) + (W.t() * psi_inv) @ W  # (k, k)

        # log|Σ| = log|M| + Σ log_ψ_i
        _, log_det_M = torch.linalg.slogdet(M)
        log_det_sigma = log_det_M + log_psi.sum()

        # Woodbury quadratic: quad_i = x_i^T Ψ^{-1} x_i - v_i^T M^{-1} v_i
        #   where v_i = W^T Ψ^{-1} x_i
        psi_inv_Xt = Xt * psi_inv                              # (n, p)
        quad1 = (Xt * psi_inv_Xt).sum(dim=1)                   # (n,)
        V = W.t() @ psi_inv_Xt.t()                             # (k, n)
        M_inv_V = torch.linalg.solve(M, V)                     # (k, n)
        quad2 = (V * M_inv_V).sum(dim=0)                       # (n,)
        quad = quad1 - quad2                                    # (n,)

        mean_nll = 0.5 * (quad.mean() + log_det_sigma + p * log_2pi)
        mean_nll.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            cur_ll = -float(mean_nll.item())
            delta = abs(cur_ll - prev_ll)
            _log.debug("FA k=%d  step=%d  mean_ll=%.6f  delta=%.2e", k, step + 1, cur_ll, delta)
            if delta < tol:
                n_iter = step + 1
                break
            prev_ll = cur_ll

    W_np = W.detach().cpu().numpy().astype(float)
    log_psi_np = log_psi.detach().cpu().numpy().astype(float)
    mean_ll = -float(mean_nll.item())
    return W_np, log_psi_np, mean_ll, n_iter


def _bic(total_log_ll: float, n: int, p: int, k: int) -> float:
    n_params = p * k + p
    return -2.0 * total_log_ll + n_params * np.log(n)


def _select_k_bic(
    Xc: np.ndarray,
    grid: list[int],
    max_iter: int,
    lr: float,
    tol: float,
    device: str,
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray, float, int]:
    """BIC sweep — returns (best_k, W, log_psi, mean_ll, n_iter) for best k."""
    n, p = Xc.shape
    best_k = grid[0]
    best_bic = float("inf")
    best_result = None

    for k in grid:
        k_eff = max(1, min(k, p - 1, n - 1))
        W, log_psi, mean_ll, n_iter = _fit_fa_woodbury(Xc, k_eff, max_iter, lr, tol, device, seed)
        total_ll = mean_ll * n
        bic = _bic(total_ll, n, p, k_eff)
        _log.info("BIC sweep k=%d  bic=%.2f  mean_ll=%.4f", k_eff, bic, mean_ll)

        if bic < best_bic:
            best_bic = bic
            best_k = k_eff
            best_result = (W, log_psi, mean_ll, n_iter)
        else:
            # First time BIC worsens — elbow found; stop.
            _log.info("BIC elbow at k=%d (first increase); selected k=%d", k_eff, best_k)
            break

    _log.info("BIC selected n_components=%d (bic=%.2f)", best_k, best_bic)
    return best_k, *best_result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# FA denoising (posterior mean reconstruction)
# ---------------------------------------------------------------------------

def _denoise(Xc: np.ndarray, W: np.ndarray, log_psi: np.ndarray) -> np.ndarray:
    """Compute denoised X via FA posterior mean E[z|x] = M^{-1} W^T Ψ^{-1} x_c.

    Never forms p×p matrices; O(n·p·k + k³).
    """
    n, p = Xc.shape
    k = W.shape[1]
    psi_inv = np.exp(-log_psi)                               # (p,)
    M = np.eye(k) + (W.T * psi_inv) @ W                     # (k, k)

    psi_inv_Xc = Xc * psi_inv                               # (n, p)
    WtPinvXc = W.T @ psi_inv_Xc.T                           # (k, n)
    M_inv_WtPinvXc = np.linalg.solve(M, WtPinvXc)           # (k, n)
    return (W @ M_inv_WtPinvXc).T                            # (n, p) denoised centred


# ---------------------------------------------------------------------------
# GpuLatentNetworkModel
# ---------------------------------------------------------------------------

@dataclass
class GpuLatentNetworkModel(NetworkModel):
    config: GpuLatentModelConfig

    def _partial_correlation(self, matrix: np.ndarray) -> np.ndarray:
        """LedoitWolf partial correlations for small p (p < 2000)."""
        if matrix.size == 0:
            return np.zeros((0, 0))
        sample_matrix = matrix.T
        covariance = LedoitWolf().fit(sample_matrix).covariance_
        precision = np.linalg.inv(covariance)
        diagonal = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
        partial = -precision / np.outer(diagonal, diagonal)
        np.fill_diagonal(partial, 0.0)
        return partial

    @staticmethod
    def _partial_correlation_from_fa(W: np.ndarray, log_psi: np.ndarray) -> np.ndarray:
        """Partial correlations via Woodbury identity — O(p²k) not O(p³).

        Σ^{-1} = Ψ^{-1} - Ψ^{-1}W (I_k + W^TΨ^{-1}W)^{-1} W^TΨ^{-1}
        Avoids LedoitWolf and matrix inversion on the p×p covariance.
        """
        p, k = W.shape
        psi_inv = np.exp(-log_psi)                             # (p,)
        A = W * psi_inv[:, None]                               # Ψ^{-1}W,  (p, k)
        M = np.eye(k) + W.T @ A                                # (k, k)
        M_inv = np.linalg.inv(M)                               # (k, k)
        B = A @ M_inv                                          # (p, k)
        # Off-diagonal precision: -B @ A.T
        prec = -(B @ A.T)                                      # (p, p)
        prec_diag = psi_inv - np.einsum("ij,ij->i", B, A)     # diagonal terms
        np.fill_diagonal(prec, prec_diag)
        scale = np.sqrt(np.clip(prec_diag, 1e-12, None))
        partial = -prec / np.outer(scale, scale)
        np.fill_diagonal(partial, 0.0)
        return partial

    def _module_table(self, graph: nx.Graph) -> pd.DataFrame:
        rows = []
        for module_index, nodes in enumerate(
            sorted(nx.connected_components(graph), key=len, reverse=True)
        ):
            if len(nodes) < self.config.min_module_size:
                continue
            for gene_id in sorted(nodes):
                rows.append({"gene_id": gene_id, "module_id": f"M{module_index:03d}"})
        return pd.DataFrame(rows)

    def _trait_associations(
        self,
        module_table: pd.DataFrame,
        feature_scores: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sample_ids = sample_table["sample_id"].tolist() if "sample_id" in sample_table.columns else list(range(len(sample_table)))
        if module_table.empty:
            return (
                pd.DataFrame(columns=["module_id", "trait", "effect", "pvalue"]),
                pd.DataFrame(columns=["module_id"] + sample_ids),
            )
        rows = []
        eigengene_rows: dict[str, list] = {}
        for module_id in sorted(module_table["module_id"].unique()):
            genes = module_table.loc[module_table["module_id"] == module_id, "gene_id"]
            subset = feature_scores.loc[feature_scores["gene_id"].isin(genes)]
            eigengene = subset.drop(columns=["gene_id"]).to_numpy(dtype=float).mean(axis=0)
            eigengene_rows[module_id] = eigengene.tolist()
            if "Age" in sample_table.columns:
                effect, pvalue = stats.pearsonr(
                    eigengene, sample_table["Age"].to_numpy(dtype=float)
                )
                rows.append({"module_id": module_id, "trait": "Age", "effect": effect, "pvalue": pvalue})
            if "Dx" in sample_table.columns:
                dx = (sample_table["Dx"] == "SCZD").astype(float).to_numpy()
                ttest = stats.ttest_ind(eigengene[dx == 0], eigengene[dx == 1], equal_var=False)
                effect = float(eigengene[dx == 1].mean() - eigengene[dx == 0].mean())
                rows.append({"module_id": module_id, "trait": "Dx", "effect": effect, "pvalue": float(ttest.pvalue)})
        eigengene_table = pd.DataFrame(eigengene_rows, index=sample_ids).T.reset_index().rename(columns={"index": "module_id"})
        return pd.DataFrame(rows), eigengene_table

    def fit(
        self,
        transcript_counts: np.ndarray,
        transcript_table: pd.DataFrame,
        sample_table: pd.DataFrame,
    ) -> FitArtifacts:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the gpu_latent backend. "
                "Install it with: pip install torch"
            )

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _log.info("GpuLatentNetworkModel using device=%s", device)

        switch_matrix, feature_info = gene_switch_coordinates(transcript_counts, transcript_table)
        if switch_matrix.size:
            design = build_design_matrix(sample_table, self.config.residualize_covariates)
            switch_matrix = residualize_rows(switch_matrix, design)

        n_genes = switch_matrix.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(feature_info["gene_id"].tolist())

        calibration: dict = {
            "mean_log_likelihood": None,
            "reconstruction_rmse": None,
            "mean_noise_variance": None,
            "gpu_latent_per_gene_noise_var": None,
            "n_components_used": 0,
            "n_components_selected_by": "none",
            "n_iter": 0,
        }
        edge_rows: list[dict] = []

        if n_genes >= 2:
            X = switch_matrix.T                   # (n_samples, n_genes)
            n, p = X.shape
            mean = X.mean(axis=0)
            Xc = X - mean

            if self.config.bic_selection and self.config.n_components_grid:
                k, W, log_psi, mean_ll, n_iter = _select_k_bic(
                    Xc,
                    self.config.n_components_grid,
                    self.config.max_iter,
                    self.config.lr,
                    self.config.tol,
                    device,
                    self.config.random_state,
                )
                selected_by = "bic"
            else:
                k = max(1, min(self.config.n_components, p - 1, n - 1))
                W, log_psi, mean_ll, n_iter = _fit_fa_woodbury(
                    Xc, k, self.config.max_iter, self.config.lr, self.config.tol,
                    device, self.config.random_state,
                )
                selected_by = "fixed"

            Xc_denoised = _denoise(Xc, W, log_psi)            # (n, p)
            X_denoised = Xc_denoised + mean
            denoised_switch = X_denoised.T                     # (p, n)

            rmse = float(np.sqrt(np.mean((Xc - Xc_denoised) ** 2)))
            noise_var = np.exp(log_psi)                        # (p,)

            calibration = {
                "mean_log_likelihood": mean_ll,
                "reconstruction_rmse": rmse,
                "mean_noise_variance": float(noise_var.mean()),
                "gpu_latent_per_gene_noise_var": noise_var.tolist(),
                "n_components_used": k,
                "n_components_selected_by": selected_by,
                "n_iter": n_iter,
            }

            if n_genes >= 2000:
                # Woodbury precision: O(p²k) vs O(p³) for LedoitWolf + inversion.
                partial = self._partial_correlation_from_fa(W, log_psi)
            else:
                partial = self._partial_correlation(denoised_switch)

            if self.config.alpha_percentile is not None:
                upper = np.abs(partial[np.triu_indices(n_genes, k=1)])
                threshold = float(np.percentile(upper, self.config.alpha_percentile))
                _log.info(
                    "auto alpha: percentile=%.1f → threshold=%.6f (fixed alpha=%.4f)",
                    self.config.alpha_percentile, threshold, self.config.alpha,
                )
                calibration["alpha_threshold"] = threshold
                calibration["alpha_percentile"] = self.config.alpha_percentile
            else:
                threshold = self.config.alpha

            # Vectorized edge extraction — avoids O(p²) Python loop.
            gene_id_arr = np.asarray(feature_info["gene_id"])
            rows_idx, cols_idx = np.triu_indices(n_genes, k=1)
            weights = partial[rows_idx, cols_idx]
            mask = np.abs(weights) >= threshold
            if mask.any():
                sel_rows = rows_idx[mask]
                sel_cols = cols_idx[mask]
                sel_w = weights[mask].tolist()
                sources = gene_id_arr[sel_rows].tolist()
                targets = gene_id_arr[sel_cols].tolist()
                graph.add_weighted_edges_from(zip(sources, targets, sel_w))
                edge_rows.extend(
                    {"source": s, "target": t, "weight": w}
                    for s, t, w in zip(sources, targets, sel_w)
                )

        module_table = self._module_table(graph)
        feature_scores = pd.DataFrame(switch_matrix, index=feature_info["gene_id"])
        feature_scores = feature_scores.reset_index().rename(columns={"index": "gene_id"})
        trait_table, eigengene_table = self._trait_associations(module_table, feature_scores, sample_table)

        return FitArtifacts(
            module_table=module_table,
            edge_table=pd.DataFrame(edge_rows),
            trait_table=trait_table,
            feature_scores=feature_scores,
            calibration=calibration,
            eigengene_table=eigengene_table,
        )
