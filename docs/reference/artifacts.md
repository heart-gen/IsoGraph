# Outputs and Artifacts

IsoGraph writes most outputs into `artifacts/`, `benchmarks/`, or `snapshots/`.

## Dataset Bundles

A prepared dataset bundle contains:

- `manifest.json`
- a sample metadata parquet file
- one or more feature-table parquet files
- one or more dense matrix `.npz` files
- optional truth tables for synthetic fixtures

## Benchmark Outputs

`benchmark` writes:

- per-fixture artifact directories under `artifacts/benchmarks/<stage_name>/<dataset>/`
- benchmark summary JSON under `artifacts/reports/<stage_name>-benchmark.json`
- runtime and memory JSON under `artifacts/reports/<stage_name>-runtime-memory.json`
- calibration JSON when supported by the selected backend
- stability-selection JSON for real-data fixtures when enabled

Each benchmark artifact directory also receives a `run.log` and a snapshot-like set of
tables written through `save_snapshot`.

For multiplex benchmarks, the benchmark JSON additionally includes `role_recall` with
per-role recall values for `switch_only`, `abundance_only`, `coupled`, and `discordant`
truth genes. WGCNA calibration reports include soft-threshold diagnostics when available.

Generated per-fixture artifact directories and generated dataset bundles are reproducible
and ignored by git in this repository. Durable benchmark evidence should be kept as compact
JSON reports under `artifacts/reports/`.

## Fit Outputs

`fit` writes:

- `modules.parquet` — module assignment per gene (`gene_id`, `module_id`)
- `edges.parquet` — inferred gene-gene edges with weights
- `traits.parquet` — module–trait associations (Pearson r or Welch t, p value, BH q value)
- `feature_scores.parquet` — per-feature switch and abundance scores; the `feature_type`
  column is `"switch"` for isoform-switch features and `"abundance"` for abundance
  features. Single-channel (switch-only) fits populate `"switch"` for all rows.
- `module_gene_roles.parquet` — gene channel role classification for each module gene:
  - `module_id`, `gene_id`
  - `module_role`: one of `coupled`, `abundance_only`, `switch_only`, `discordant`, `inactive`
  - `switch_r` — Pearson r between this gene's switch feature and the module eigengene
  - `abundance_r` — Pearson r between this gene's abundance feature and the module eigengene
  - `switch_abundance_r` — Pearson r between this gene's switch and abundance features
  - `switch_active`, `abundance_active` — boolean flags (|r| ≥ 0.2)
- `fit_config.json` — complete configuration used for this run, including IsoGraph version
  and random seed

## Explain Outputs

`explain-module` writes per module (by default under `artifacts/explain/<study>/`):

- `gene_driver_table.parquet` — genes ranked by |r| with the module eigengene; columns
  include `gene_id`, `r`, `pvalue`, `qvalue`, `n_complete`, `missingness`, `feature_id`,
  `feature_type` (for multiplex: `"switch"` or `"abundance"`)
- `transcript_polarity_table.parquet` — transcript-level correlations with module
  eigengene; includes `switch_strength = max(r) − min(r)` per gene
- `high_vs_low_table.parquet` — Welch test contrasts between high- and low-eigengene
  samples per feature
- `vae_drivers.parquet` (when `--vae-attribution`) — high-confidence VAE decoder
  attribution; columns include `gene_id`, `decoded_response`, `feature_type`
- `ig_attributions.parquet` (when `--integrated-gradients`) — Captum Integrated
  Gradients encoder attribution; columns include `gene_id`, `mean_ig`, `mean_abs_ig`,
  `latent_dim`, `latent_eigengene_r`
- `module_explanation_manifest.json` — index of all output files for this module

## Compare Outputs

`compare` writes a JSON report describing either:

- differences between two snapshot directories, or
- delta metrics between two benchmark summary files

## Export Outputs

`export` writes a JSON summary of a prepared dataset bundle, including sample count, gene
count, and available assays.
