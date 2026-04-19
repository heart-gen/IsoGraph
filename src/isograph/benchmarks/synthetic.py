"""Synthetic benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from isograph.io.artifacts import (
    DatasetBundle,
    build_feature_spec,
    build_matrix_spec,
    save_dataset_bundle,
)
from isograph.validation import DatasetManifest


# ---------------------------------------------------------------------------
# Simple (idealized) fixtures — toy_v1 and medium_v1
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDatasetSpec:
    name: str
    n_genes: int
    n_samples: int
    n_modules: int
    seed: int


def _module_assignments(n_genes: int, n_modules: int) -> np.ndarray:
    modules = np.repeat(np.arange(n_modules), np.ceil(n_genes / n_modules))[:n_genes]
    return modules.astype(int)


def _generate_dataset(spec: SyntheticDatasetSpec, suite_name: str, description: str) -> DatasetBundle:
    rng = np.random.default_rng(spec.seed)
    sample_ids = [f"S{i:03d}" for i in range(spec.n_samples)]
    gene_ids = [f"G{i:04d}" for i in range(spec.n_genes)]
    module_ids = _module_assignments(spec.n_genes, spec.n_modules)

    dx = np.array(["Control"] * (spec.n_samples // 2) + ["SCZD"] * (spec.n_samples - spec.n_samples // 2))
    age = np.linspace(30, 70, spec.n_samples) + rng.normal(0, 2, spec.n_samples)
    sex = np.where(np.arange(spec.n_samples) % 2 == 0, "F", "M")
    sample_table = pd.DataFrame({"sample_id": sample_ids, "Dx": dx, "Age": age, "Sex": sex})

    module_latent = rng.normal(size=(spec.n_modules, spec.n_samples))
    module_latent += (dx == "SCZD")[None, :] * np.linspace(0.8, 1.2, spec.n_modules)[:, None]
    module_latent += ((age - age.mean()) / age.std())[None, :] * np.linspace(0.4, 0.7, spec.n_modules)[:, None]

    gene_totals = rng.gamma(shape=10.0, scale=30.0, size=(spec.n_genes, spec.n_samples))
    switch_signal = np.vstack([module_latent[module_ids[index]] for index in range(spec.n_genes)])
    p1 = 1.0 / (1.0 + np.exp(-switch_signal))
    p2 = 1.0 - p1
    transcript_counts = np.zeros((spec.n_genes * 2, spec.n_samples), dtype=float)
    transcript_feature_rows: list[dict[str, object]] = []
    for index, gene_id in enumerate(gene_ids):
        total = np.maximum(gene_totals[index], 1.0)
        tx1 = np.floor(total * p1[index])
        tx2 = np.maximum(total - tx1, 0.0)
        transcript_counts[index * 2] = tx1
        transcript_counts[index * 2 + 1] = tx2
        transcript_feature_rows.extend(
            [
                {"transcript_id": f"{gene_id}_T1", "gene_id": gene_id, "length": 1000},
                {"transcript_id": f"{gene_id}_T2", "gene_id": gene_id, "length": 950},
            ]
        )

    gene_counts = transcript_counts.reshape(spec.n_genes, 2, spec.n_samples).sum(axis=1)
    psi = np.clip(p1 + rng.normal(0, 0.03, size=p1.shape), 1e-4, 1 - 1e-4)
    gene_table = pd.DataFrame(
        {"gene_id": gene_ids, "chrom": "chrSynthetic", "start": np.arange(spec.n_genes), "end": np.arange(spec.n_genes) + 100}
    )
    transcript_table = pd.DataFrame(transcript_feature_rows)
    psi_table = pd.DataFrame({"psi_uid": [f"PSI_{gene_id}" for gene_id in gene_ids], "gene_id": gene_ids})
    truth_modules = pd.DataFrame({"gene_id": gene_ids, "module_id": module_ids})
    truth_switch = pd.DataFrame({"gene_id": gene_ids, "has_switch": True})

    manifest = DatasetManifest(
        dataset_name=spec.name,
        suite_name=suite_name,
        description=description,
        sample_table="samples.parquet",
        feature_tables=[
            build_feature_spec("gene", "genes.parquet", gene_table),
            build_feature_spec("transcript", "transcripts.parquet", transcript_table),
            build_feature_spec("psi", "psi.parquet", psi_table),
            build_feature_spec("truth_module", "truth_modules.parquet", truth_modules),
            build_feature_spec("truth_switch", "truth_switch.parquet", truth_switch),
        ],
        matrices=[
            build_matrix_spec("gene_counts", "gene_counts.npz", gene_counts),
            build_matrix_spec("transcript_counts", "transcript_counts.npz", transcript_counts),
            build_matrix_spec("psi", "psi.npz", psi),
        ],
        provenance={"generator": "synthetic_v1", "seed": str(spec.seed)},
        truth_tables=["truth_modules.parquet", "truth_switch.parquet"],
    )
    return DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables={
            "gene": gene_table,
            "transcript": transcript_table,
            "psi": psi_table,
            "truth_module": truth_modules,
            "truth_switch": truth_switch,
        },
        matrices={"gene_counts": gene_counts, "transcript_counts": transcript_counts, "psi": psi},
        truth_tables={"truth_modules.parquet": truth_modules, "truth_switch.parquet": truth_switch},
    )


# ---------------------------------------------------------------------------
# Realistic fixtures — realistic_v1 and realistic_unequal_v1
#
# These fixtures are designed to stress-test assumptions that toy_v1 and
# medium_v1 do not exercise:
#
#   1. Non-switching background genes (~50 % of all genes) — the model must
#      identify modules from a noisy background without ground truth labels.
#   2. Variable isoform counts (2–5 per gene, sampled from a realistic
#      distribution) — the CLR / SVD pipeline must handle higher-dimensional
#      composition spaces.
#   3. Negative-binomial overdispersion (Gamma-Poisson mixture) — more
#      realistic than Poisson-like floor counts.
#   4. Between-module partial correlation via a shared latent confounder —
#      modules are no longer perfectly orthogonal.
#
# The two fixtures differ only in module-size distribution:
#   realistic_v1          — equal-sized modules (isolates features 1–4 above)
#   realistic_unequal_v1  — power-law-sized modules (adds size imbalance)
# ---------------------------------------------------------------------------

@dataclass
class RealisticDatasetSpec:
    name: str
    n_genes: int
    n_samples: int
    n_modules: int
    # None → equal-sized modules; explicit list → custom sizes (must sum to n_switching)
    module_sizes: list[int] | None = None
    switching_fraction: float = 0.5
    confounder_weight: float = 0.3
    count_dispersion: float = 5.0
    mean_gene_total: float = 300.0
    seed: int = 0
    # Effect size ranges: uniform draw per module. Default matches original realistic fixtures.
    dx_effect_range: tuple[float, float] = (0.5, 1.5)
    age_effect_range: tuple[float, float] = (0.2, 0.6)
    # Dirichlet concentration for isoform proportions.
    # Lower switching_concentration → noisier isoform splits within switching genes.
    switching_concentration: float = 20.0
    nonswitching_concentration: float = 100.0


# Isoform-count distribution: 2=45 %, 3=30 %, 4=15 %, 5=10 %
_ISOFORM_CHOICES = np.array([2, 3, 4, 5])
_ISOFORM_PROBS   = np.array([0.45, 0.30, 0.15, 0.10])


def _equal_module_sizes(n_switching: int, n_modules: int) -> list[int]:
    base = n_switching // n_modules
    sizes = [base] * n_modules
    sizes[0] += n_switching - sum(sizes)
    return sizes


def _nb_counts(
    rng: np.random.Generator, mean: float, dispersion: float, size: int
) -> np.ndarray:
    """Negative-binomial via Gamma-Poisson mixture. Supports float dispersion."""
    rates = rng.gamma(shape=dispersion, scale=mean / dispersion, size=size)
    return rng.poisson(rates).astype(float)


def _dirichlet_props(
    rng: np.random.Generator, alpha: np.ndarray
) -> np.ndarray:
    """Vectorised Dirichlet draw.  alpha shape: (k, n_samples) → output (k, n_samples)."""
    g = rng.gamma(shape=np.maximum(alpha, 1e-6), scale=1.0)
    return g / g.sum(axis=0, keepdims=True)


def _generate_realistic_dataset(
    spec: RealisticDatasetSpec, suite_name: str, description: str
) -> DatasetBundle:
    rng = np.random.default_rng(spec.seed)
    n_samples = spec.n_samples
    n_genes = spec.n_genes

    # ------------------------------------------------------------------
    # Sample metadata
    # ------------------------------------------------------------------
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    n_ctrl = n_samples // 2
    dx  = np.array(["Control"] * n_ctrl + ["SCZD"] * (n_samples - n_ctrl))
    age = rng.normal(50, 12, n_samples).clip(25, 85)
    sex = rng.choice(["F", "M"], size=n_samples)
    sample_table = pd.DataFrame(
        {"sample_id": sample_ids, "Dx": dx, "Age": age, "Sex": sex}
    )

    # ------------------------------------------------------------------
    # Gene partitioning: switching vs non-switching
    # ------------------------------------------------------------------
    n_switching    = int(round(n_genes * spec.switching_fraction))
    n_nonswitching = n_genes - n_switching
    gene_ids       = [f"G{i:04d}" for i in range(n_genes)]

    if spec.module_sizes is not None:
        if sum(spec.module_sizes) != n_switching:
            raise ValueError(
                f"module_sizes sum ({sum(spec.module_sizes)}) != n_switching ({n_switching})"
            )
        sizes = spec.module_sizes
    else:
        sizes = _equal_module_sizes(n_switching, spec.n_modules)

    # Module assignment for switching genes (sequential within each module)
    module_assignments = np.repeat(np.arange(spec.n_modules), sizes)

    # ------------------------------------------------------------------
    # Isoform counts per gene
    # ------------------------------------------------------------------
    n_isoforms = rng.choice(_ISOFORM_CHOICES, size=n_genes, p=_ISOFORM_PROBS)

    # ------------------------------------------------------------------
    # Module latents with shared confounder
    # ------------------------------------------------------------------
    module_latent = rng.normal(size=(spec.n_modules, n_samples))
    # Dx effect (different magnitude per module)
    dx_effects = rng.uniform(*spec.dx_effect_range, spec.n_modules)
    module_latent += (dx == "SCZD")[None, :] * dx_effects[:, None]
    # Age effect
    age_z = (age - age.mean()) / age.std()
    age_effects = rng.uniform(*spec.age_effect_range, spec.n_modules)
    module_latent += age_z[None, :] * age_effects[:, None]
    # Shared confounder → between-module partial correlations
    z_confound = rng.normal(size=n_samples)
    module_latent += spec.confounder_weight * z_confound[None, :]

    # ------------------------------------------------------------------
    # Transcript count generation
    # ------------------------------------------------------------------
    all_tx_rows: list[dict] = []
    all_tx_counts: list[np.ndarray] = []

    truth_module_rows: list[dict] = []
    truth_switch_rows: list[dict] = []

    for gene_idx, gene_id in enumerate(gene_ids):
        k     = int(n_isoforms[gene_idx])
        is_sw = gene_idx < n_switching

        # NB total reads for this gene across all samples
        total = _nb_counts(rng, spec.mean_gene_total, spec.count_dispersion, n_samples)
        total = np.maximum(total, k)  # at least 1 read per isoform

        if is_sw:
            mod = int(module_assignments[gene_idx])
            # Isoform 1 proportion driven by module signal via sigmoid
            p1 = 1.0 / (1.0 + np.exp(-module_latent[mod]))  # (n_samples,)
            if k == 2:
                raw = np.stack([p1, 1.0 - p1], axis=0)  # (2, n_samples)
            else:
                p_rest = (1.0 - p1) / (k - 1)           # (n_samples,)
                raw = np.vstack([p1[None, :], np.tile(p_rest[None, :], (k - 1, 1))])
            # Add Dirichlet noise (concentration controls isoform proportion noise)
            props = _dirichlet_props(rng, raw * spec.switching_concentration)

            truth_switch_rows.append({"gene_id": gene_id, "has_switch": True})
            truth_module_rows.append({"gene_id": gene_id, "module_id": mod})
        else:
            # Non-switching: stable gene-specific proportions + small noise
            base = rng.dirichlet(np.ones(k) * 3)          # (k,) fixed per gene
            alpha = np.tile(base[:, None], (1, n_samples)) * spec.nonswitching_concentration
            props = _dirichlet_props(rng, alpha)

            truth_switch_rows.append({"gene_id": gene_id, "has_switch": False})

        # Proportion → integer counts, last isoform absorbs rounding residual
        tx = np.zeros((k, n_samples), dtype=float)
        for j in range(k - 1):
            tx[j] = np.floor(total * props[j])
        tx[k - 1] = np.maximum(total - tx[: k - 1].sum(axis=0), 0.0)

        all_tx_counts.append(tx)
        for j in range(k):
            all_tx_rows.append({
                "transcript_id": f"{gene_id}_T{j + 1}",
                "gene_id": gene_id,
                "length": 1000 - j * 50,
            })

    transcript_counts = np.vstack(all_tx_counts)        # (total_tx, n_samples)
    transcript_table  = pd.DataFrame(all_tx_rows)

    # ------------------------------------------------------------------
    # Derived feature matrices
    # ------------------------------------------------------------------
    gene_counts = np.zeros((n_genes, n_samples))
    psi         = np.zeros((n_genes, n_samples))
    tx_offset   = 0
    for gene_idx in range(n_genes):
        k = int(n_isoforms[gene_idx])
        block = transcript_counts[tx_offset : tx_offset + k]
        total = block.sum(axis=0)
        gene_counts[gene_idx] = total
        psi_raw = block[0] / np.maximum(total, 1)
        psi[gene_idx] = np.clip(
            psi_raw + rng.normal(0, 0.02, n_samples), 1e-4, 1 - 1e-4
        )
        tx_offset += k

    # ------------------------------------------------------------------
    # Tables and truth
    # ------------------------------------------------------------------
    gene_table = pd.DataFrame({
        "gene_id": gene_ids,
        "chrom":   "chrRealistic",
        "start":   np.arange(n_genes) * 1000,
        "end":     np.arange(n_genes) * 1000 + 500,
    })
    psi_table = pd.DataFrame({
        "psi_uid": [f"PSI_{g}" for g in gene_ids],
        "gene_id": gene_ids,
    })
    truth_modules_df = pd.DataFrame(truth_module_rows)
    truth_switch_df  = pd.DataFrame(truth_switch_rows)

    # ------------------------------------------------------------------
    # Bundle
    # ------------------------------------------------------------------
    manifest = DatasetManifest(
        dataset_name=spec.name,
        suite_name=suite_name,
        description=description,
        sample_table="samples.parquet",
        feature_tables=[
            build_feature_spec("gene",         "genes.parquet",           gene_table),
            build_feature_spec("transcript",   "transcripts.parquet",     transcript_table),
            build_feature_spec("psi",          "psi.parquet",             psi_table),
            build_feature_spec("truth_module", "truth_modules.parquet",   truth_modules_df),
            build_feature_spec("truth_switch", "truth_switch.parquet",    truth_switch_df),
        ],
        matrices=[
            build_matrix_spec("gene_counts",       "gene_counts.npz",       gene_counts),
            build_matrix_spec("transcript_counts", "transcript_counts.npz", transcript_counts),
            build_matrix_spec("psi",               "psi.npz",               psi),
        ],
        provenance={
            "generator":          "realistic_v1",
            "seed":               str(spec.seed),
            "switching_fraction": str(spec.switching_fraction),
            "confounder_weight":  str(spec.confounder_weight),
            "count_dispersion":   str(spec.count_dispersion),
            "module_sizes":       str(sizes),
        },
        truth_tables=["truth_modules.parquet", "truth_switch.parquet"],
    )
    return DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables={
            "gene":         gene_table,
            "transcript":   transcript_table,
            "psi":          psi_table,
            "truth_module": truth_modules_df,
            "truth_switch": truth_switch_df,
        },
        matrices={
            "gene_counts":       gene_counts,
            "transcript_counts": transcript_counts,
            "psi":               psi,
        },
        truth_tables={
            "truth_modules.parquet": truth_modules_df,
            "truth_switch.parquet":  truth_switch_df,
        },
    )


# ---------------------------------------------------------------------------
# Nonlinear fixture — nonlinear_v1
#
# Module activation is driven by quadrant membership in a 2-D latent space
# (f1 × f2 interaction), not a linear combination of factors.  This breaks
# the FA linear-Gaussian assumption: FA can identify f1 and f2 as separate
# components but cannot represent the product f1·f2 that defines each module.
#
# The four modules activate in:
#   M0 → f1 > 0  AND  f2 > 0   (SCZD-high, Age-high)
#   M1 → f1 < 0  AND  f2 > 0   (Control,   Age-high)
#   M2 → f1 > 0  AND  f2 < 0   (SCZD-high, Age-low)
#   M3 → f1 < 0  AND  f2 < 0   (Control,   Age-low)
#
# Linear methods score each sample on f1 and f2 independently, so they
# predict M0 genes as loading on f1 + f2 (incorrect) and cannot recover
# the within-quadrant structure cleanly.  A VAE with a non-linear decoder
# can model the joint threshold non-linearity.
# ---------------------------------------------------------------------------


@dataclass
class NonlinearDatasetSpec:
    name: str
    n_genes: int
    n_samples: int
    n_modules: int = 4            # must be a multiple of 4 for quadrant assignment
    module_sizes: list[int] | None = None
    switching_fraction: float = 0.5
    count_dispersion: float = 5.0
    mean_gene_total: float = 300.0
    state_effect_size: float = 2.5   # within-quadrant signal strength
    state_background: float = 0.15   # cross-quadrant noise standard deviation
    confounder_weight: float = 0.2
    switching_concentration: float = 20.0
    nonswitching_concentration: float = 100.0
    seed: int = 0


def _generate_nonlinear_dataset(
    spec: NonlinearDatasetSpec, suite_name: str, description: str
) -> DatasetBundle:
    rng = np.random.default_rng(spec.seed)
    n_samples = spec.n_samples
    n_genes = spec.n_genes

    # ------------------------------------------------------------------
    # Sample metadata
    # ------------------------------------------------------------------
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]
    n_ctrl = n_samples // 2
    dx  = np.array(["Control"] * n_ctrl + ["SCZD"] * (n_samples - n_ctrl))
    age = rng.normal(50, 12, n_samples).clip(25, 85)
    sex = rng.choice(["F", "M"], size=n_samples)
    sample_table = pd.DataFrame(
        {"sample_id": sample_ids, "Dx": dx, "Age": age, "Sex": sex}
    )

    # ------------------------------------------------------------------
    # Two latent factors correlated with Dx and Age
    # ------------------------------------------------------------------
    f1 = rng.normal(0, 0.8, n_samples)
    f1 += (dx == "SCZD").astype(float) * 1.5   # SCZD pushes f1 positive

    age_z = (age - age.mean()) / age.std()
    f2 = rng.normal(0, 0.8, n_samples)
    f2 += age_z * 1.2                            # older samples push f2 positive

    # Shared confounder (induces residual between-module correlation)
    z_confound = rng.normal(size=n_samples)

    # ------------------------------------------------------------------
    # Non-linear module activation via radial / product structure.
    #
    # Modules cycle through four non-linear patterns for n_modules > 4:
    #   m % 4 == 0 → inner ring: r² < P35 (cannot be expressed as a1·f1 + a2·f2)
    #   m % 4 == 1 → outer ring: r² > P65 (same)
    #   m % 4 == 2 → positive product: f1·f2 > P65(|f1·f2|)
    #   m % 4 == 3 → negative product: f1·f2 < -P65(|f1·f2|)
    #
    # None of these can be written as a linear combination of f1 and f2.
    # FA recovers f1 and f2 individually but cannot represent r² or f1·f2.
    # ------------------------------------------------------------------
    n_switching = int(round(n_genes * spec.switching_fraction))
    n_nonswitching = n_genes - n_switching
    gene_ids = [f"G{i:04d}" for i in range(n_genes)]

    if spec.module_sizes is not None:
        if sum(spec.module_sizes) != n_switching:
            raise ValueError(
                f"module_sizes sum ({sum(spec.module_sizes)}) != n_switching ({n_switching})"
            )
        sizes = spec.module_sizes
    else:
        sizes = _equal_module_sizes(n_switching, spec.n_modules)

    module_assignments = np.repeat(np.arange(spec.n_modules), sizes)

    r_sq   = f1**2 + f2**2
    prod   = f1 * f2
    p35_r  = float(np.percentile(r_sq, 35))
    p65_r  = float(np.percentile(r_sq, 65))
    p65_p  = float(np.percentile(np.abs(prod), 65))

    nonlinear_masks = [
        r_sq < p35_r,          # inner ring
        r_sq > p65_r,          # outer ring
        prod > p65_p,          # strong positive product  (SCZD–old or Control–young)
        prod < -p65_p,         # strong negative product  (SCZD–young or Control–old)
    ]

    module_latent = np.zeros((spec.n_modules, n_samples))
    for m in range(spec.n_modules):
        active = nonlinear_masks[m % 4].astype(float)
        module_latent[m] = (
            active * spec.state_effect_size
            + rng.normal(0, spec.state_background, n_samples)
            + spec.confounder_weight * z_confound
        )

    # ------------------------------------------------------------------
    # Isoform counts per gene
    # ------------------------------------------------------------------
    n_isoforms = rng.choice(_ISOFORM_CHOICES, size=n_genes, p=_ISOFORM_PROBS)

    # ------------------------------------------------------------------
    # Transcript count generation (identical structure to realistic)
    # ------------------------------------------------------------------
    all_tx_rows: list[dict] = []
    all_tx_counts: list[np.ndarray] = []
    truth_module_rows: list[dict] = []
    truth_switch_rows: list[dict] = []

    for gene_idx, gene_id in enumerate(gene_ids):
        k     = int(n_isoforms[gene_idx])
        is_sw = gene_idx < n_switching

        total = _nb_counts(rng, spec.mean_gene_total, spec.count_dispersion, n_samples)
        total = np.maximum(total, k)

        if is_sw:
            mod = int(module_assignments[gene_idx])
            p1  = 1.0 / (1.0 + np.exp(-module_latent[mod]))
            if k == 2:
                raw = np.stack([p1, 1.0 - p1], axis=0)
            else:
                p_rest = (1.0 - p1) / (k - 1)
                raw = np.vstack([p1[None, :], np.tile(p_rest[None, :], (k - 1, 1))])
            props = _dirichlet_props(rng, raw * spec.switching_concentration)
            truth_switch_rows.append({"gene_id": gene_id, "has_switch": True})
            truth_module_rows.append({"gene_id": gene_id, "module_id": mod})
        else:
            base  = rng.dirichlet(np.ones(k) * 3)
            alpha = np.tile(base[:, None], (1, n_samples)) * spec.nonswitching_concentration
            props = _dirichlet_props(rng, alpha)
            truth_switch_rows.append({"gene_id": gene_id, "has_switch": False})

        tx = np.zeros((k, n_samples), dtype=float)
        for j in range(k - 1):
            tx[j] = np.floor(total * props[j])
        tx[k - 1] = np.maximum(total - tx[: k - 1].sum(axis=0), 0.0)

        all_tx_counts.append(tx)
        for j in range(k):
            all_tx_rows.append({
                "transcript_id": f"{gene_id}_T{j + 1}",
                "gene_id": gene_id,
                "length": 1000 - j * 50,
            })

    transcript_counts = np.vstack(all_tx_counts)
    transcript_table  = pd.DataFrame(all_tx_rows)

    # ------------------------------------------------------------------
    # Derived feature matrices
    # ------------------------------------------------------------------
    gene_counts = np.zeros((n_genes, n_samples))
    psi         = np.zeros((n_genes, n_samples))
    tx_offset   = 0
    for gene_idx in range(n_genes):
        k     = int(n_isoforms[gene_idx])
        block = transcript_counts[tx_offset : tx_offset + k]
        total = block.sum(axis=0)
        gene_counts[gene_idx] = total
        psi_raw = block[0] / np.maximum(total, 1)
        psi[gene_idx] = np.clip(
            psi_raw + rng.normal(0, 0.02, n_samples), 1e-4, 1 - 1e-4
        )
        tx_offset += k

    # ------------------------------------------------------------------
    # Tables and truth
    # ------------------------------------------------------------------
    gene_table = pd.DataFrame({
        "gene_id": gene_ids,
        "chrom":   "chrNonlinear",
        "start":   np.arange(n_genes) * 1000,
        "end":     np.arange(n_genes) * 1000 + 500,
    })
    psi_table = pd.DataFrame({
        "psi_uid": [f"PSI_{g}" for g in gene_ids],
        "gene_id": gene_ids,
    })
    truth_modules_df = pd.DataFrame(truth_module_rows)
    truth_switch_df  = pd.DataFrame(truth_switch_rows)

    manifest = DatasetManifest(
        dataset_name=spec.name,
        suite_name=suite_name,
        description=description,
        sample_table="samples.parquet",
        feature_tables=[
            build_feature_spec("gene",         "genes.parquet",           gene_table),
            build_feature_spec("transcript",   "transcripts.parquet",     transcript_table),
            build_feature_spec("psi",          "psi.parquet",             psi_table),
            build_feature_spec("truth_module", "truth_modules.parquet",   truth_modules_df),
            build_feature_spec("truth_switch", "truth_switch.parquet",    truth_switch_df),
        ],
        matrices=[
            build_matrix_spec("gene_counts",       "gene_counts.npz",       gene_counts),
            build_matrix_spec("transcript_counts", "transcript_counts.npz", transcript_counts),
            build_matrix_spec("psi",               "psi.npz",               psi),
        ],
        provenance={
            "generator":        "nonlinear_v1",
            "seed":             str(spec.seed),
            "state_effect_size": str(spec.state_effect_size),
            "n_modules":        str(spec.n_modules),
        },
        truth_tables=["truth_modules.parquet", "truth_switch.parquet"],
    )
    return DatasetBundle(
        manifest=manifest,
        sample_table=sample_table,
        feature_tables={
            "gene":         gene_table,
            "transcript":   transcript_table,
            "psi":          psi_table,
            "truth_module": truth_modules_df,
            "truth_switch": truth_switch_df,
        },
        matrices={
            "gene_counts":       gene_counts,
            "transcript_counts": transcript_counts,
            "psi":               psi,
        },
        truth_tables={
            "truth_modules.parquet": truth_modules_df,
            "truth_switch.parquet":  truth_switch_df,
        },
    )


# ---------------------------------------------------------------------------
# Core suite
# ---------------------------------------------------------------------------

def generate_core_suite(root: Path, seed: int) -> list[Path]:
    suite_dir = root / "core_v1"

    toy = _generate_dataset(
        SyntheticDatasetSpec(name="toy_v1", n_genes=24, n_samples=48, n_modules=2, seed=seed),
        suite_name="core_v1",
        description="Toy deterministic simulation with exact switch truth.",
    )
    medium = _generate_dataset(
        SyntheticDatasetSpec(
            name="medium_v1", n_genes=400, n_samples=240, n_modules=8, seed=seed + 1
        ),
        suite_name="core_v1",
        description="Medium deterministic simulation with planted switch modules.",
    )

    # realistic_v1: equal-sized modules — isolates non-switching background,
    # variable isoform counts, NB overdispersion, and between-module confounder.
    realistic = _generate_realistic_dataset(
        RealisticDatasetSpec(
            name="realistic_v1",
            n_genes=200,
            n_samples=160,
            n_modules=5,
            module_sizes=None,   # equal: [20, 20, 20, 20, 20]
            switching_fraction=0.5,
            confounder_weight=0.3,
            count_dispersion=5.0,
            mean_gene_total=300.0,
            seed=seed + 2,
        ),
        suite_name="core_v1",
        description=(
            "Realistic simulation: 50 % non-switching genes, variable isoform counts "
            "(2–5), NB overdispersion, shared confounder. Equal-sized modules."
        ),
    )

    # realistic_unequal_v1: power-law module sizes — adds module-size imbalance
    # on top of all the realistic features in realistic_v1.
    realistic_unequal = _generate_realistic_dataset(
        RealisticDatasetSpec(
            name="realistic_unequal_v1",
            n_genes=200,
            n_samples=160,
            n_modules=5,
            module_sizes=[38, 25, 17, 12, 8],   # power-law, sum=100
            switching_fraction=0.5,
            confounder_weight=0.3,
            count_dispersion=5.0,
            mean_gene_total=300.0,
            seed=seed + 3,
        ),
        suite_name="core_v1",
        description=(
            "Realistic simulation: 50 % non-switching genes, variable isoform counts "
            "(2–5), NB overdispersion, shared confounder. Power-law module sizes "
            "[38, 25, 17, 12, 8]."
        ),
    )

    # ------------------------------------------------------------------
    # noisy_v1: High-noise stress test grounded in real caudate parameters.
    #
    # Motivation: real_caudate_aa_v1 has median NB dispersion ~6 and shows
    # reconstruction RMSE ~1.9 vs ~0.36 for realistic_v1, indicating ~5×
    # more noise.  noisy_v1 pushes dispersion to 15, reduces sample count
    # to 100, weakens module effects, and raises background to 70 % to
    # create a fixture where Stage 1 fails and Stage 2 / Stage 3 can
    # demonstrate improvement.
    #
    # Module sizes [22,18,14,11,9,8,7,6] are power-law (sum=95 ≈ 30 %
    # switching), consistent with the sparse switching signal seen in real
    # data where only a fraction of genes show isoform shifts.
    # ------------------------------------------------------------------
    noisy = _generate_realistic_dataset(
        RealisticDatasetSpec(
            name="noisy_v1",
            n_genes=300,
            n_samples=100,
            n_modules=8,
            module_sizes=[22, 18, 14, 11, 9, 8, 7, 6],   # power-law, sum=95
            switching_fraction=95 / 300,                   # ≈ 31.7 %
            confounder_weight=0.6,
            count_dispersion=15.0,
            mean_gene_total=300.0,
            dx_effect_range=(0.2, 0.6),
            age_effect_range=(0.1, 0.25),
            switching_concentration=10.0,   # noisier isoform splits
            nonswitching_concentration=80.0,
            seed=seed + 4,
        ),
        suite_name="core_v1",
        description=(
            "High-noise stress test: 300 genes, 100 samples, 8 power-law modules, "
            "~70 % background, NB dispersion=15, weak Dx/Age effects (0.2–0.6 / "
            "0.1–0.25), strong confounder (weight=0.6). Grounded in real caudate "
            "dispersion (~median 6, max 25) and sparse switching fraction."
        ),
    )

    # ------------------------------------------------------------------
    # large_v1: Scale stress test (n_genes >> n_samples).
    #
    # 800 genes / 120 samples → ratio 6.7×.  With Ledoit-Wolf, Stage 1
    # estimates an 800×800 covariance from 120 observations — severely
    # under-determined.  FA (Stage 2 / 3) reduces the effective problem to
    # k × k (k ≈ 10 modules) where estimation is tractable.
    #
    # Parameters match real caudate dispersion (7.0 ≈ median 5.98) and
    # approximate the realistic switching fraction (~25 %).  Module sizes
    # follow a power-law to match typical WGCNA-style outputs from brain
    # tissue: a few large "hub" modules and many small specialised ones.
    # ------------------------------------------------------------------
    large = _generate_realistic_dataset(
        RealisticDatasetSpec(
            name="large_v1",
            n_genes=800,
            n_samples=120,
            n_modules=10,
            module_sizes=[40, 32, 25, 20, 18, 16, 15, 13, 12, 9],  # power-law, sum=200
            switching_fraction=200 / 800,   # 25 %
            confounder_weight=0.4,
            count_dispersion=7.0,
            mean_gene_total=300.0,
            dx_effect_range=(0.4, 0.9),
            age_effect_range=(0.15, 0.45),
            switching_concentration=20.0,
            nonswitching_concentration=100.0,
            seed=seed + 5,
        ),
        suite_name="core_v1",
        description=(
            "Scale stress test: 800 genes / 120 samples (ratio 6.7×), 10 power-law "
            "modules, 75 % background, NB dispersion=7 (real caudate median). "
            "Stage 1 covariance estimation is severely underdetermined; FA "
            "dimension reduction (Stage 2 / 3) is the key differentiator."
        ),
    )

    # ------------------------------------------------------------------
    # nonlinear_v1: Non-linear module structure (quadrant interaction).
    #
    # Module activation is determined by the PRODUCT of two latent factors
    # (f1 correlated with Dx, f2 correlated with Age), not their sum:
    #   M0 active when f1 > 0 AND f2 > 0  (SCZD + older)
    #   M1 active when f1 < 0 AND f2 > 0  (Control + older)
    #   M2 active when f1 > 0 AND f2 < 0  (SCZD + younger)
    #   M3 active when f1 < 0 AND f2 < 0  (Control + younger)
    #
    # FA (Stage 2 / 3) extracts f1 and f2 as orthogonal components but
    # cannot represent the threshold product f1·f2 that defines each
    # module — leading to partial or incorrect module assignments.  A VAE
    # with a non-linear encoder/decoder (Stage 4) should handle this.
    # ------------------------------------------------------------------
    nonlinear = _generate_nonlinear_dataset(
        NonlinearDatasetSpec(
            name="nonlinear_v1",
            n_genes=200,
            n_samples=200,
            n_modules=4,
            module_sizes=[30, 25, 25, 20],   # slightly unequal, sum=100
            switching_fraction=0.5,
            count_dispersion=5.0,
            mean_gene_total=300.0,
            state_effect_size=2.5,
            state_background=0.15,
            confounder_weight=0.2,
            switching_concentration=20.0,
            nonswitching_concentration=100.0,
            seed=seed + 6,
        ),
        suite_name="core_v1",
        description=(
            "Non-linear stress test: 4 modules with radial / product-interaction "
            "structure (inner ring, outer ring, positive f1·f2, negative f1·f2). "
            "None of these can be expressed as a1·f1 + a2·f2, so linear FA / "
            "partial-correlation methods cannot fully recover the modules. "
            "Designed as the Stage 4 (VAE) target fixture."
        ),
    )

    return [
        save_dataset_bundle(toy,               suite_dir / "toy_v1"),
        save_dataset_bundle(medium,            suite_dir / "medium_v1"),
        save_dataset_bundle(realistic,         suite_dir / "realistic_v1"),
        save_dataset_bundle(realistic_unequal, suite_dir / "realistic_unequal_v1"),
        save_dataset_bundle(noisy,             suite_dir / "noisy_v1"),
        save_dataset_bundle(large,             suite_dir / "large_v1"),
        save_dataset_bundle(nonlinear,         suite_dir / "nonlinear_v1"),
    ]
