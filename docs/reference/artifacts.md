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
- `traits.parquet` — module–trait associations, one row per module × trait, with columns
  `module_id`, `trait`, `effect`, `pvalue`. `effect` is the Pearson r for numeric traits and
  the Welch group-mean difference for binary categorical traits. `pvalue` is **uncorrected** —
  no q value is written; apply FDR control downstream.
- `feature_scores.parquet` — per-feature switch and abundance scores; the `feature_type`
  column is `"switch"` for isoform-switch features and `"abundance"` for abundance
  features. Single-channel (switch-only) fits populate `"switch"` for all rows.
  On the `vae` backend these are the **raw, pre-residualization** values even when
  `residualize_covariates` is set; the other backends persist the residualized matrix — see
  [Residualization is discovery-only](configuration.md#residualization-is-a-discovery-only-knob-vae-backend).
- `module_gene_roles.parquet` — gene channel role classification for each module gene:
  - `module_id`, `gene_id`
  - `module_role`: one of `coupled`, `abundance_only`, `switch_only`, `discordant`, `inactive`
  - `switch_r` — Pearson r between this gene's switch feature and the module eigengene
  - `abundance_r` — Pearson r between this gene's abundance feature and the module eigengene
  - `switch_abundance_r` — Pearson r between this gene's switch and abundance features
  - `switch_active`, `abundance_active` — boolean flags (|r| ≥ 0.2)
- `fit_config.json` — complete configuration used for this run, including IsoGraph version
  and random seed
- `calibration.json` — backend calibration metadata when the selected backend emits it
  (VAE, latent). For VAE this includes `reconstruction_rmse`, `vae_latent_dim`, and
  `vae_n_collapsed_dims`; multiplex runs additionally record the auto-selected thresholds
  (`selected_alpha_switch`, `selected_alpha_abundance`) and `leiden_selection`.

```{note}
`FitArtifacts` also carries `node_diagnostics`, `feature_reconstruction`, and
`residualization_qc` diagnostic tables in-process. These are available from the Python API
(`model.fit(...)`) but are not written to disk by `isograph fit`.
```

### `residualization_qc`

Populated on `FitArtifacts` by the `vae` backend only, whenever `residualize_covariates`
resolves to at least one usable covariate (it is `None` otherwise, and on every other
backend). One row per residualized feature, with the identifier columns
available in the feature table (`feature_id`, `gene_id`, `feature_type`) plus:

- `var_before`, `var_after` — per-feature variance before and after residualization
- `var_retained_frac` — `var_after / var_before`; how much signal survived. Values near 0
  mean the covariates absorbed nearly all of a feature's variance.
- `confound_r2_before`, `confound_r2_after` — fraction of the feature's variance explained
  by the covariate axis, before and after.

A working residualization drives `confound_r2_after` toward zero; `var_retained_frac` is the
collateral cost. This is the diagnostic that is observable on real data, where module
recovery is not.

```python
artifacts = model.fit(...)
qc = artifacts.residualization_qc
qc["confound_r2_after"].max()        # should be ~0
qc["var_retained_frac"].median()     # how much signal the covariates cost
```

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
