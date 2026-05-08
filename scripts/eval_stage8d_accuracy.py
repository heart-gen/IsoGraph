"""Stage 8D accuracy evaluation: VAE decoder attribution vs ground truth.

Runs for each fixture:
  - toy_v1, medium_v1, realistic_v1, noisy_v1, nonlinear_v1, large_v1,
    xlarge_mini, xxlarge_mini

Metrics:
  switch_auc_vae   -- AUC(|decoded_delta|, truth_switch) over module genes
  switch_auc_r     -- AUC(|r|, truth_switch) over module genes (8A baseline)
  member_auc       -- AUC(|decoded_delta|, module_membership) over ALL genes
  sign_agree       -- fraction of module genes where sign(decoded_delta)==sign(r)
  precision@filter -- among passes_filter=True genes, fraction that are true switches

For toy_v1 / medium_v1 (all genes switch), switch_auc is uninformative (all
positive); member_auc is the primary metric.

Usage:
    python scripts/eval_stage8d_accuracy.py
"""

from __future__ import annotations

import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isograph.benchmarks.synthetic import (
    NonlinearDatasetSpec,
    RealisticDatasetSpec,
    SyntheticDatasetSpec,
    _generate_dataset,
    _generate_nonlinear_dataset,
    _generate_realistic_dataset,
)
from isograph.explain.config import ExplainConfig
from isograph.explain.core import explain_module
from isograph.explain.vae_attribution import compute_decoder_jacobian
from isograph.features.channels import feature_sample_columns
from isograph.models.vae import VaeNetworkModel
from isograph.workflow.config import VaeModelConfig


# ---------------------------------------------------------------------------
# Fixture specs
# ---------------------------------------------------------------------------

_FIXTURES = [
    ("toy_v1",       "simple",   SyntheticDatasetSpec(name="toy_v1",    n_genes=24,  n_samples=48,  n_modules=2, seed=0)),
    ("medium_v1",    "simple",   SyntheticDatasetSpec(name="medium_v1", n_genes=400, n_samples=240, n_modules=8, seed=1)),
    ("realistic_v1", "realistic", RealisticDatasetSpec(
        name="realistic_v1", n_genes=200, n_samples=160, n_modules=5,
        module_sizes=None, switching_fraction=0.5, confounder_weight=0.3,
        count_dispersion=5.0, mean_gene_total=300.0, seed=2,
    )),
    ("noisy_v1", "realistic", RealisticDatasetSpec(
        name="noisy_v1", n_genes=300, n_samples=100, n_modules=8,
        module_sizes=[22, 18, 14, 11, 9, 8, 7, 6], switching_fraction=95/300,
        confounder_weight=0.6, count_dispersion=15.0, mean_gene_total=300.0,
        dx_effect_range=(0.2, 0.6), age_effect_range=(0.1, 0.25),
        switching_concentration=10.0, nonswitching_concentration=80.0, seed=4,
    )),
    ("nonlinear_v1", "nonlinear", NonlinearDatasetSpec(
        name="nonlinear_v1", n_genes=200, n_samples=200, n_modules=4,
        n_nonlinear_modules=4, module_sizes=[30, 25, 25, 20],
        switching_fraction=0.5, count_dispersion=5.0, mean_gene_total=300.0,
        state_effect_size=2.5, state_background=0.15, confounder_weight=0.2,
        switching_concentration=20.0, nonswitching_concentration=100.0, seed=42,
    )),
    ("large_v1", "realistic", RealisticDatasetSpec(
        name="large_v1", n_genes=800, n_samples=120, n_modules=10,
        module_sizes=[70, 55, 42, 32, 25, 20, 15, 12, 10, 9],  # power-law sum=290
        switching_fraction=290/800, confounder_weight=0.4,
        count_dispersion=7.0, mean_gene_total=300.0,
        dx_effect_range=(0.3, 0.8), age_effect_range=(0.1, 0.35),
        switching_concentration=15.0, nonswitching_concentration=80.0, seed=5,
    )),
    # xlarge/xxlarge at 1/10 scale to keep runtime reasonable
    ("xlarge_mini", "realistic", RealisticDatasetSpec(
        name="xlarge_mini", n_genes=600, n_samples=240, n_modules=12,
        module_sizes=None, switching_fraction=84/600,
        confounder_weight=0.4, count_dispersion=7.0, mean_gene_total=300.0,
        dx_effect_range=(0.4, 0.9), age_effect_range=(0.15, 0.45),
        switching_concentration=20.0, nonswitching_concentration=100.0, seed=42,
    )),
    ("xxlarge_mini", "realistic", RealisticDatasetSpec(
        name="xxlarge_mini", n_genes=400, n_samples=240, n_modules=16,
        module_sizes=None, switching_fraction=96/400,
        confounder_weight=0.4, count_dispersion=7.0, mean_gene_total=300.0,
        dx_effect_range=(0.4, 0.9), age_effect_range=(0.15, 0.45),
        switching_concentration=20.0, nonswitching_concentration=100.0, seed=42,
    )),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bundle(spec):
    if isinstance(spec, SyntheticDatasetSpec):
        return _generate_dataset(spec, "eval8d", spec.name)
    if isinstance(spec, NonlinearDatasetSpec):
        return _generate_nonlinear_dataset(spec, "eval8d", spec.name)
    return _generate_realistic_dataset(spec, "eval8d", spec.name)


def _build_transcript_usage(transcript_counts, transcript_table, sample_ids):
    n_tx = len(transcript_table)
    usage = np.zeros((n_tx, len(sample_ids)), dtype=float)
    for _, grp in transcript_table.groupby("gene_id", sort=False):
        idx = grp.index.to_numpy()
        block = transcript_counts[idx]
        totals = block.sum(axis=0, keepdims=True)
        usage[idx] = block / np.maximum(totals, 1.0)
    return pd.DataFrame(usage.T, index=sample_ids, columns=transcript_table["transcript_id"].tolist())


def _vae_config_for(n_genes: int, artifact_dir: Path) -> VaeModelConfig:
    """Pick reasonable VAE config for fixture size."""
    if n_genes <= 100:
        hidden_dim, n_hidden = 32, 1
    elif n_genes <= 500:
        hidden_dim, n_hidden = 64, 1
    elif n_genes <= 1000:
        hidden_dim, n_hidden = 128, 2
    else:
        hidden_dim, n_hidden = 256, 2
    return VaeModelConfig(
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
        latent_dim_grid=[2, 4, 6, 8],
        alpha=0.70,
        random_state=42,
        checkpoint_dir=artifact_dir,
    )


def _fit_vae_and_write(bundle, artifact_dir: Path):
    sample_ids = bundle.sample_table["sample_id"].tolist()
    n_genes = len(bundle.feature_tables["transcript"]["gene_id"].unique())
    cfg = _vae_config_for(n_genes, artifact_dir)
    model = VaeNetworkModel(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifacts = model.fit(
            transcript_counts=bundle.matrices["transcript_counts"],
            transcript_table=bundle.feature_tables["transcript"],
            sample_table=bundle.sample_table,
    )
    # Rename integer columns to sample IDs
    fs = artifacts.feature_scores.copy()
    sample_cols = feature_sample_columns(fs)
    fs = fs.rename(columns={old: new for old, new in zip(sample_cols, sample_ids)})
    artifacts.module_table.to_parquet(artifact_dir / "modules.parquet", index=False)
    fs.to_parquet(artifact_dir / "feature_scores.parquet", index=False)
    return artifacts.module_table, fs


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    count = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(count / (len(pos) * len(neg)))


def _collapse_decoder_by_gene(jacobian: pd.DataFrame) -> pd.DataFrame:
    """Keep the strongest attribution row per gene for gene-level metrics."""
    return (
        jacobian.assign(_abs_delta=jacobian["decoded_delta"].abs())
        .sort_values("_abs_delta", ascending=False)
        .drop_duplicates("gene_id", keep="first")
        .drop(columns=["_abs_delta"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

# Modules with fewer than this many genes are excluded from member_auc_large.
# Tiny spurious modules (2-4 genes) from over-segmentation produce sub-random
# member_auc that contaminates the mean — see nonlinear_v1 diagnosis.
_MIN_MODULE_GENES_LARGE = 10


@dataclass
class AccuracyResult:
    fixture: str
    n_genes: int
    n_samples: int
    n_fitted_modules: int
    switch_auc_vae: float   # AUC(|decoded_delta|, truth_switch) over module genes
    switch_auc_r: float     # AUC(|r|, truth_switch) over module genes
    member_auc: float       # AUC(|decoded_delta|, module_membership) over ALL genes
    member_auc_large: float # member_auc restricted to modules with ≥_MIN_MODULE_GENES_LARGE genes
    sign_agree: float       # fraction sign(delta)==sign(r)
    precision_at_filter: float  # precision among passes_filter=True genes
    fit_seconds: float
    attr_seconds: float
    all_switch: bool        # True if fixture has no non-switching genes (toy/medium)


def _compute_accuracy(
    fixture_name: str,
    bundle,
    artifact_dir: Path,
    explain_results: dict,
    fit_seconds: float,
) -> AccuracyResult:
    truth_switch = bundle.feature_tables["truth_switch"]
    switch_set = set(truth_switch.loc[truth_switch["has_switch"], "gene_id"])
    all_switch = bool(truth_switch["has_switch"].all())

    checkpoint_path = artifact_dir / "vae_checkpoint.pt"
    feature_scores = pd.read_parquet(artifact_dir / "feature_scores.parquet")
    n_genes = len(feature_scores)

    t0 = time.time()

    # --- Per-module Jacobians (all genes) ---
    jacobians: dict[str, pd.DataFrame] = {}
    for module_id, result in explain_results.items():
        jacobians[module_id] = compute_decoder_jacobian(
            checkpoint_path, result.eigengene, feature_scores
        )

    attr_seconds = time.time() - t0

    # --- Metric 1: switch_auc over module genes ---
    vae_scores, vae_labels = [], []
    r_scores, r_labels = [], []

    for module_id, result in explain_results.items():
        jac = jacobians[module_id]
        jac_gene = _collapse_decoder_by_gene(jac)
        jac_idx = jac_gene.set_index("gene_id")["decoded_delta"]
        tbl = result.gene_driver_table
        finite = tbl["r"].notna()
        for _, row in tbl.loc[finite].iterrows():
            gid = row["gene_id"]
            label = 1 if gid in switch_set else 0
            r_scores.append(abs(row["r"]))
            r_labels.append(label)
            if gid in jac_idx.index:
                vae_scores.append(abs(jac_idx[gid]))
                vae_labels.append(label)

    switch_auc_vae = _auc(np.array(vae_scores), np.array(vae_labels)) if vae_scores else float("nan")
    switch_auc_r = _auc(np.array(r_scores), np.array(r_labels)) if r_scores else float("nan")

    # --- Metric 2: module membership AUC over all genes ---
    member_aucs = []
    member_aucs_large = []
    for module_id, result in explain_results.items():
        jac = _collapse_decoder_by_gene(jacobians[module_id])
        module_genes = set(result.gene_driver_table["gene_id"].tolist())
        labels = np.array([1 if g in module_genes else 0 for g in jac["gene_id"]])
        scores = jac["decoded_delta"].abs().values
        if labels.sum() > 0 and (labels == 0).sum() > 0:
            auc_val = _auc(scores, labels)
            member_aucs.append(auc_val)
            if len(module_genes) >= _MIN_MODULE_GENES_LARGE:
                member_aucs_large.append(auc_val)
    member_auc = float(np.mean(member_aucs)) if member_aucs else float("nan")
    member_auc_large = float(np.mean(member_aucs_large)) if member_aucs_large else float("nan")

    # --- Metric 3: sign agreement (direction-corrected by latent_r sign) ---
    sign_agreements = []
    for module_id, result in explain_results.items():
        jac = _collapse_decoder_by_gene(jacobians[module_id]).set_index("gene_id")
        tbl = result.gene_driver_table
        finite = tbl["r"].notna()
        for _, row in tbl.loc[finite].iterrows():
            gid = row["gene_id"]
            if gid in jac.index and not np.isnan(jac.loc[gid, "decoded_delta"]):
                latent_r = jac.loc[gid, "latent_r"]
                direction = np.sign(latent_r) if latent_r != 0 else 1.0
                corrected = jac.loc[gid, "decoded_delta"] * direction
                agree = np.sign(corrected) == np.sign(row["r"])
                sign_agreements.append(int(agree))
    sign_agree = float(np.mean(sign_agreements)) if sign_agreements else float("nan")

    # --- Metric 4: precision@filter ---
    config = ExplainConfig(vae_attribution=False, vae_fdr_threshold=0.05, vae_percentile_threshold=90.0)
    precision_vals = []
    if not all_switch:  # Only meaningful when non-switching genes exist
        from isograph.explain.vae_attribution import filter_vae_drivers
        for module_id, result in explain_results.items():
            jac = jacobians[module_id]
            vae_df = filter_vae_drivers(jac, result.gene_driver_table, config)
            passing = vae_df.loc[vae_df["passes_filter"], "gene_id"].tolist()
            if passing:
                prec = sum(1 for g in passing if g in switch_set) / len(passing)
                precision_vals.append(prec)
    precision_at_filter = float(np.mean(precision_vals)) if precision_vals else float("nan")

    n_samples = len(bundle.sample_table)
    return AccuracyResult(
        fixture=fixture_name,
        n_genes=n_genes,
        n_samples=n_samples,
        n_fitted_modules=len(explain_results),
        switch_auc_vae=switch_auc_vae,
        switch_auc_r=switch_auc_r,
        member_auc=member_auc,
        member_auc_large=member_auc_large,
        sign_agree=sign_agree,
        precision_at_filter=precision_at_filter,
        fit_seconds=fit_seconds,
        attr_seconds=attr_seconds,
        all_switch=all_switch,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    results: list[AccuracyResult] = []

    for fixture_name, fixture_type, spec in _FIXTURES:
        n_genes = spec.n_genes
        n_samples = spec.n_samples
        print(f"\n{'='*60}")
        print(f"  {fixture_name}  ({n_genes} genes, {n_samples} samples)")
        print(f"{'='*60}")

        print("  Generating bundle...", end=" ", flush=True)
        bundle = _make_bundle(spec)
        print("done")

        sample_ids = bundle.sample_table["sample_id"].tolist()
        transcript_table = bundle.feature_tables["transcript"]

        feature_table = _build_transcript_usage(
            bundle.matrices["transcript_counts"], transcript_table, sample_ids
        )
        feature_meta = pd.DataFrame({
            "feature_id": transcript_table["transcript_id"].tolist(),
            "gene_id": transcript_table["gene_id"].tolist(),
            "transcript_id": transcript_table["transcript_id"].tolist(),
            "feature_type": "transcript_usage",
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)

            print("  Fitting VAE...", end=" ", flush=True)
            t0 = time.time()
            module_table, _ = _fit_vae_and_write(bundle, artifact_dir)
            fit_seconds = time.time() - t0
            print(f"{fit_seconds:.1f}s  ({len(module_table['module_id'].unique())} modules)")

            if module_table.empty:
                print("  !! No modules fitted — skipping")
                continue

            print("  Running explain_module...", end=" ", flush=True)
            config = ExplainConfig(vae_attribution=False)  # attribution computed separately below
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                explain_results = explain_module(
                    artifact_dir=artifact_dir,
                    feature_table=feature_table,
                    feature_meta=feature_meta,
                    config=config,
                )
            print(f"done ({len(explain_results)} modules)")

            print("  Computing VAE attribution accuracy...", end=" ", flush=True)
            result = _compute_accuracy(
                fixture_name, bundle, artifact_dir, explain_results, fit_seconds
            )
            print(f"done ({result.attr_seconds:.2f}s)")

        results.append(result)
        _print_row(result)

    return results


def _print_row(r: AccuracyResult):
    tag = " (all-switch)" if r.all_switch else ""
    print(f"\n  Results for {r.fixture}{tag}:")
    print(f"    Fitted modules:    {r.n_fitted_modules}")
    print(f"    member_auc:        {r.member_auc:.3f}   (|decoded_delta| vs module membership, all genes)")
    if not np.isnan(r.member_auc_large):
        print(f"    member_auc_large:  {r.member_auc_large:.3f}   (modules with ≥{_MIN_MODULE_GENES_LARGE} genes only)")
    print(f"    switch_auc_vae:    {r.switch_auc_vae:.3f}   (|decoded_delta| vs truth_switch, module genes)")
    print(f"    switch_auc_r:      {r.switch_auc_r:.3f}   (|r| vs truth_switch — 8A baseline)")
    print(f"    sign_agree:        {r.sign_agree:.3f}   (fraction sign(delta)==sign(r))")
    if not np.isnan(r.precision_at_filter):
        print(f"    precision@filter:  {r.precision_at_filter:.3f}   (true switching among passes_filter=True)")
    print(f"    fit_time:          {r.fit_seconds:.1f}s")
    print(f"    attr_time:         {r.attr_seconds:.2f}s")


def _print_summary(results: list[AccuracyResult]):
    print("\n\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    hdr = f"{'Fixture':<16} {'Genes':>6} {'Mods':>5}  {'member_auc':>10}  {'mem_large':>9}  {'switch_vae':>10}  {'switch_r':>10}  {'sign_agree':>10}  {'prec@filt':>9}"
    print(hdr)
    print("-"*100)
    for r in results:
        tag = "*" if r.all_switch else " "
        ml = f"{r.member_auc_large:>9.3f}" if not np.isnan(r.member_auc_large) else f"{'—':>9}"
        row = (
            f"{r.fixture:<16}{tag}"
            f"{r.n_genes:>6}  {r.n_fitted_modules:>4}  "
            f"  {r.member_auc:>10.3f}"
            f"  {ml}"
            f"  {r.switch_auc_vae:>10.3f}"
            f"  {r.switch_auc_r:>10.3f}"
            f"  {r.sign_agree:>10.3f}"
            f"  {r.precision_at_filter:>9.3f}"
        )
        print(row)
    print("-"*100)
    print("  * = all genes are switching (toy/medium); switch_auc uninformative")
    print("  member_auc: AUC of |decoded_delta| distinguishing module genes from all other genes")
    print("  switch_vae: AUC of |decoded_delta| predicting truth_switch, within module genes")
    print("  switch_r:   AUC of |r| predicting truth_switch (Stage 8A baseline)")
    print("  sign_agree: fraction of module genes where sign(decoded_delta) == sign(r)")
    print("  prec@filt:  precision of passes_filter=True (FDR≤0.05 & top-10% & sign agree)")


if __name__ == "__main__":
    results = run_all()
    _print_summary(results)
