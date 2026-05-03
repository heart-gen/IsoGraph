# CLI Reference

IsoGraph exposes a single CLI entry point:

```bash
isograph --help
```

The current subcommands are `benchmark`, `freeze-real`, `fit`, `compare`, `export`,
`explain-module`, and `annotate-structure`.

## Overrides

`benchmark`, `freeze-real`, and `fit` accept Hydra-style overrides after `--`:

```bash
isograph benchmark -- backend=latent fixture_filter=medium_v1 stage_name=stage2_docs
isograph fit --dataset-path my_cohort --backend vae -- vae.alpha=0.6 vae.hidden_dim=256
```

## `benchmark`

Run the bundled fixture suite, or a filtered subset, through a selected backend.

The default backend is `vae`. Available backends: `baseline`, `latent`, `graph`, `vae`,
`wgcna`.

Examples:

```bash
# VAE on a single fixture (default backend)
isograph benchmark -- fixture_filter=toy_v1 stage_name=vae_toy

# WGCNA on the scale suite
isograph benchmark --config-name stage6_scale_comparison_wgcna
```

Behavior:

- For `dataset_suite: core_v1` — generates the synthetic `core_v1` fixtures as needed and
  freezes the real fixture unless `fixture_filter` targets only synthetic datasets.
- For `dataset_suite: scale_v1` — generates `xlarge_v1` (6 000 genes), `xxlarge_v1`
  (12 000 genes), and `xxlarge_stress_v1` (12 000 genes, stressed parameters).
- Writes per-fixture artifacts under `artifacts/benchmarks/<stage_name>/`.
- Writes benchmark and runtime summaries under `artifacts/reports/`.
- Writes calibration reports when the selected backend emits calibration metadata.

## `freeze-real`

Freeze the bundled real-data fixture from local source tables.

Example:

```bash
isograph freeze-real --suite-name core_v1
```

The command reads `BenchmarkCommandConfig.real_data` through the benchmark config and
caches intermediate selections under `benchmarks/cache/real_data/`.

## `fit`

Fit any backend on a prepared dataset bundle. VAE is the default.

```bash
# VAE (default)
isograph fit \
  --dataset-path benchmarks/datasets/custom/my_cohort_v1 \
  --output-dir artifacts/fits/vae_default

# Baseline
isograph fit \
  --dataset-path benchmarks/datasets/core_v1/toy_v1 \
  --backend baseline \
  --output-dir artifacts/fits/toy_v1

# VAE with Hydra overrides
isograph fit \
  --dataset-path benchmarks/datasets/custom/my_cohort_v1 \
  --backend vae \
  --output-dir artifacts/fits/vae_tuned \
  -- vae.alpha=0.6 vae.hidden_dim=256 vae.n_epochs=400
```

Available backends: `baseline`, `latent`, `graph`, `vae`, `wgcna`.

Outputs:

- `modules.parquet`
- `edges.parquet`
- `traits.parquet`
- `feature_scores.parquet`
- `calibration.json` (when the backend emits calibration metadata — VAE, latent)
- `fit_config.json`

Default config values for all backends live in `configs/fit.yaml` and can be
overridden with Hydra syntax after `--`.

## `explain-module`

Explain one or more fitted modules at transcript-feature resolution.

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --module-ids M000 M001 \
  --output-dir artifacts/explain/run1
```

With plots and VAE decoder attribution:

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --plot --output-format png pdf \
  --vae-attribution \
  --output-dir artifacts/explain/run1
```

With Captum Integrated Gradients (requires `pip install isograph[torch-explain]`):

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --integrated-gradients --ig-n-steps 100 \
  --output-dir artifacts/explain/run1
```

With a structural annotation table (from `annotate-structure`):

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --annotation-table transcript_structure_annotations.tsv \
  --output-dir artifacts/explain/run1
```

**Inputs:**

- `--artifact-dir` must contain `modules.parquet` and `feature_scores.parquet`.
- `--feature-table`: Parquet with sample IDs as index (or `sample_id` column), feature IDs
  as columns.
- `--feature-meta`: Parquet or TSV with columns `feature_id`, `gene_id`, `feature_type`
  (required); `gene_name`, `transcript_id`, `exon_id`, `event_id` (optional).

**Outputs per module** in `{output_dir}/{module_id}/`:

- `gene_driver_table.parquet` — gene-level drivers sorted by |r|
- `transcript_polarity_table.parquet` — transcript-level correlations with `switch_strength`
- `high_vs_low_table.parquet` — mean usage contrast between high- and low-module samples
- `vae_drivers.parquet` — high-confidence VAE decoder attribution (with `--vae-attribution`)
- `ig_attributions.parquet` — per-feature IG scores (with `--integrated-gradients`)
- Plot files (`*.png`, `*.pdf`) when `--plot` is given

**Shared manifest:** `{output_dir}/module_explanation_manifest.json`

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--module-ids` | all | Module IDs to explain |
| `--plot` | off | Write publication-ready plot files |
| `--output-format` | png | `png`, `pdf`, or both |
| `--annotation-table` | none | Structural annotation TSV from `annotate-structure` |
| `--vae-attribution` | off | VAE decoder Jacobian attribution (needs checkpoint) |
| `--vae-fdr-threshold` | 0.05 | FDR cutoff for high-confidence VAE drivers |
| `--vae-percentile-threshold` | 90.0 | `\|decoded_delta\|` percentile for VAE drivers |
| `--integrated-gradients` | off | Captum IG encoder attribution (needs checkpoint + captum) |
| `--ig-n-steps` | 50 | IG interpolation steps |
| `--ig-baseline` | zero | IG baseline: `zero` or `mean` |

## `annotate-structure`

Annotate transcript switch pairs with structural labels from a GTF file.

```bash
isograph annotate-structure \
  --gtf gencode.v47.annotation.gtf.gz \
  --switch-pairs switch_pairs.tsv \
  --output transcript_structure_annotations.tsv
```

Cache the parsed GTF to avoid re-parsing on repeated runs (raw GENCODE v47 ≈ 20 min on NFS;
cached ≈ seconds):

```bash
isograph annotate-structure \
  --gtf gencode.v47.annotation.gtf.gz \
  --switch-pairs switch_pairs.tsv \
  --gtf-cache gencode_v47_cache.parquet \
  --output transcript_structure_annotations.tsv
```

**Inputs:**

- `--gtf`: GTF or GTF.gz annotation file (GENCODE or Ensembl conventions supported).
- `--switch-pairs`: TSV with columns `gene_id`, `transcript_id_1`, `transcript_id_2`.

**Output** (`--output`): TSV with structural labels per switch pair:

| Label | Type | Description |
|---|---|---|
| `first_exon_changed` | bool | First exon differs between transcripts |
| `last_exon_changed` | bool | Last exon differs |
| `internal_exon_diff` | bool | Internal exon composition differs |
| `cds_changed` | bool | CDS coordinates differ |
| `utr_changed` | bool | UTR coordinates differ |
| `biotype_switch` | bool | Transcript biotype differs |
| `coding_status_change` | bool | One transcript is coding, the other is not |
| `tx_length_delta` | float | Transcript length difference (bp) |
| `cds_length_delta` | float | CDS length difference (bp) |
| `shared_exon_fraction` | float | Fraction of exons shared between the two transcripts |

Pass the output to `isograph explain-module --annotation-table` to merge these labels
into the driver tables.

## `compare`

Compare either two snapshot directories or two benchmark JSON reports.

Examples:

```bash
isograph compare \
  --reference snapshots/stage0_toy_v1_baseline_v1_seed0000 \
  --candidate artifacts/benchmarks/quickstart_baseline/toy_v1
```

```bash
isograph compare \
  --reference artifacts/reports/stage2_latent-benchmark.json \
  --candidate artifacts/reports/stage4_vae-benchmark.json
```

## `export`

Write a JSON summary of a prepared dataset bundle.

Example:

```bash
isograph export \
  --dataset-path benchmarks/datasets/core_v1/toy_v1 \
  --output-path artifacts/reports/toy_v1-summary.json
```
