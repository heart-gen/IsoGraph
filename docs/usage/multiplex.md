# Multiplex and Stress Benchmarks

Isoform switching and gene expression can change independently. A gene that doubles its
expression without changing its dominant isoform is an abundance event; a gene that
redistributes transcript usage without changing total expression is a switching event.
Both types of co-regulation occur in real RNA-seq data, and treating them as a single
signal conflates distinct biological mechanisms.

IsoGraph separates these signals by constructing two feature channels per gene: an
**abundance channel** (log-CPM–standardized gene counts, available for all genes) and a
**switch channel** (CLR-SVD transcript usage coordinate, available for multi-isoform
genes). A typed feature graph with per-channel edge thresholds and an auto-calibrated
abundance threshold connects genes through whichever channels are active. The `multiplex_v1`
benchmark suite provides planted ground-truth modules with known role assignments to
validate that IsoGraph recovers each type.

The benchmark runner reports overall planted-module recovery plus role-aware recall for:

- `switch_only` — genes whose module membership is driven by switching, not expression
- `abundance_only` — genes driven by expression level changes only
- `coupled` — both channels active and positively correlated (r ≥ 0.2)
- `discordant` — both channels active but anticorrelated (r < −0.2)

## Standard Multiplex Suite

The `multiplex_v1` suite contains four regular fixtures:

| Fixture | Genes | Samples | Planted modules |
|---|---:|---:|---:|
| `toy_multiplex_v1` | 40 | 64 | 2 |
| `medium_multiplex_v1` | 320 | 180 | 6 |
| `noisy_multiplex_v1` | 360 | 110 | 6 |
| `large_multiplex_v1` | 900 | 140 | 10 |

Run the tuned Stage 9 backends:

```bash
isograph benchmark --config-name stage9_multiplex_vae
isograph benchmark --config-name stage9_multiplex_graph
isograph benchmark --config-name stage9_multiplex_latent
isograph benchmark --config-name stage9_multiplex_wgcna
```

Aggregate stress reports against WGCNA:

```bash
python scripts/stress_multiplex_summary.py
```

The summary is written to `artifacts/reports/stress-multiplex-backend-summary.json`.

## Explain Compatibility

Explain modules work on multiplex artifacts. The attribution outputs preserve
`feature_id` and `feature_type` so a gene can be explained through its switch channel,
abundance channel, or both. The stress helper evaluates explain accuracy across the
standard multiplex stress artifacts:

```bash
python scripts/stress_multiplex_explain.py
```

The helper writes `artifacts/explain/stress_multiplex/stress_multiplex_explain_metrics.json`.

## Extra-Large Multiplex Stress Fixture

The 12k-gene fixture is intentionally generated only when requested by name so routine
test and benchmark runs do not pay its cost.

| Fixture | Genes | Samples | Planted modules |
|---|---:|---:|---:|
| `xxlarge_multiplex_v1` | 12,000 | 240 | 16 |

Run VAE and WGCNA:

```bash
isograph benchmark --config-name stress_multiplex_xxlarge_vae
isograph benchmark --config-name stress_multiplex_xxlarge_wgcna
```

The current stress reports show:

| Backend | Detected modules | Recovery | Runtime seconds | Notes |
|---|---:|---:|---:|---|
| VAE | 16 | 0.9266666666666667 | 7554.699150358792 | Correct module count; no posterior collapse |
| WGCNA | 1898 | 0.9191666666666665 | 877.2644422741141 | Severe over-segmentation |

VAE role-aware recall on `xxlarge_multiplex_v1` was `1.0` for `switch_only`, `1.0`
for `coupled`, `1.0` for `discordant`, and `0.725` for `abundance_only`.

## Understanding Role-Aware Recall

Role-aware recall measures the fraction of planted genes in each role that IsoGraph
assigns to the correct module. The calibration gates for VAE on the standard multiplex
suite are:

| Fixture | Backend | Overall recovery | `abundance_only` recall | `switch_only` recall |
|---|---|---:|---:|---:|
| `toy_multiplex_v1` | VAE | ≥ 0.40 | ≥ 0.90 | ≥ 0.90 |
| `medium_multiplex_v1` | VAE | ≥ 0.895 | ≥ 0.90 | ≥ 0.90 |
| `noisy_multiplex_v1` | VAE | ≥ 0.883 | ≥ 0.90 | ≥ 0.90 |
| `large_multiplex_v1` | VAE | ≥ 0.896 | ≥ 0.90 | ≥ 0.90 |

**Why giant-component fraction matters.** A well-calibrated run produces modules that are
clearly separated in the gene graph. When the abundance threshold is too permissive, many
genes connect into one giant component — WGCNA shows this pattern at 12k scale (1898
detected modules vs. 16 planted). Check `giant_component_fraction < 0.05` as a sanity
metric; it is reported in the benchmark JSON.

**What low recall means.** If `abundance_only` recall is low (< 0.70) on your real data,
the abundance threshold may be merging abundance modules with switch modules. Lower
`alpha_abundance` or reduce the `alpha_abundance_grid` range. If `switch_only` recall is
low, the switch threshold may be too strict; lower `alpha_switch`.

## Artifact Policy

Generated dataset bundles and per-fixture benchmark directories are reproducible and
ignored by git. Commit compact evidence as JSON reports under `artifacts/reports/`.
