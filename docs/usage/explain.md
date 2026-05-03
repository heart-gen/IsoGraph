# Module Explanation

After fitting IsoGraph on a dataset, the `isograph.explain` module identifies the
transcript-level and gene-local features that best explain each discovered module. The
key biological object is a within-gene isoform switch: a module driver should be a
directional transcript-usage contrast within a gene, not just a transcript with high
marginal correlation.

## Quick Start

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --module-ids M000 M001 \
  --plot \
  --output-dir artifacts/explain/run1
```

## Python API

```python
from isograph.explain import explain_module, ExplainConfig

results = explain_module(
    artifact_dir="artifacts/fits/my_dataset",
    feature_table=feature_df,          # samples × features DataFrame
    feature_meta=meta_df,              # feature_id, gene_id, feature_type columns
    module_ids=["M000", "M001"],
    config=ExplainConfig(plot=True),
    output_dir="artifacts/explain/run1",
)
# results["M000"].gene_driver_table — top gene drivers sorted by |r|
# results["M000"].transcript_polarity_table — per-transcript correlations
# results["M000"].high_vs_low_table — high- vs. low-module contrast
```

`explain_module` returns a `dict[str, ExplainResult]`.

## Inputs

| Input | Required | Description |
|---|---|---|
| `artifact_dir` | yes | Must contain `modules.parquet` and `feature_scores.parquet` |
| `feature_table` | yes | Samples × features DataFrame (index or `sample_id` column) |
| `feature_meta` | yes | Columns: `feature_id`, `gene_id`, `feature_type` (required); `gene_name`, `transcript_id`, `exon_id`, `event_id`, `source_coordinate` (optional) |
| `module_ids` | no | Subset of module IDs to explain (default: all modules) |
| `module_score_table` | no | Precomputed eigengene scores (overrides computed eigengenes) |
| `annotation_table` | no | Structural labels from `isograph annotate-structure` |

## Outputs

For each module, `explain-module` writes a subdirectory `{output_dir}/{module_id}/`:

### `gene_driver_table.parquet`

One row per gene. Columns: `gene_id`, `r` (Pearson correlation with module eigengene),
`pvalue`, `qvalue` (BH-corrected), `n_samples`, `missing_fraction`. Sorted by `|r|`.

### `transcript_polarity_table.parquet`

One row per feature. Columns: `feature_id`, `gene_id`, `r`, `pvalue`, `qvalue`,
`n_samples`, `missing_fraction`, `switch_strength`. `switch_strength = max(r) − min(r)`
across transcripts within a gene — high values identify genes with directional isoform
switching.

### `high_vs_low_table.parquet`

One row per feature. Contrasts mean feature usage in the top vs. bottom percentile of
samples ranked by module eigengene. Columns: `feature_id`, `gene_id`, `mean_high`,
`mean_low`, `delta`, `se`, `tstat`, `pvalue`, `n_high`, `n_low`, `missing_fraction`.

### `module_explanation_manifest.json`

Shared manifest at `{output_dir}/module_explanation_manifest.json` recording config,
module IDs, file paths, and optional feature availability.

## Publication-Ready Plots

Pass `--plot` (or `ExplainConfig(plot=True)`) to write per-module figures:

- **Top driver genes barplot** — |r| with 95% CI, colored by polarity direction
- **Transcript usage gradient** — x = module eigengene score, y = transcript usage,
  smoothed trend per transcript for the top N driver genes
- **Positive/negative driver heatmap** — heatmap of normalized usage for top positive and
  negative driver genes

Use `--output-format pdf` or `--output-format png pdf` to control format.

## Structural Annotation

`isograph annotate-structure` assigns GTF-based structural labels to transcript switch
pairs (exon changes, CDS/UTR shifts, biotype switches). Pass the output to
`--annotation-table` to merge those labels into the driver tables.

```bash
# Step 1 — annotate switch pairs
isograph annotate-structure \
  --gtf gencode.v47.annotation.gtf.gz \
  --switch-pairs switch_pairs.tsv \
  --gtf-cache gencode_v47_cache.parquet \
  --output transcript_structure_annotations.tsv

# Step 2 — explain with structural context
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --annotation-table transcript_structure_annotations.tsv \
  --output-dir artifacts/explain/run1
```

The GTF cache (written on the first run) avoids re-parsing the full GTF on subsequent
runs. Parsing GENCODE v47 from scratch takes ~20 minutes on NFS; the cache loads in
seconds.

## VAE Decoder Attribution

When the fit artifact directory contains a VAE checkpoint (`vae_checkpoint.pt`), pass
`--vae-attribution` to compute a finite-difference Jacobian attribution:

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --vae-attribution \
  --vae-fdr-threshold 0.05 \
  --vae-percentile-threshold 90.0 \
  --output-dir artifacts/explain/run1
```

This perturbs the module-associated latent dimension and decodes back into gene space via
the VAE decoder. Genes passing three simultaneous criteria are classified as
high-confidence drivers:

1. Association FDR ≤ `--vae-fdr-threshold` (from `gene_driver_table.qvalue`)
2. `|decoded_delta|` in the top `--vae-percentile-threshold` percentile among module genes
3. `sign(decoded_delta)` agrees with `sign(r)` (direction-corrected for anti-correlated latent dims)

Output: `{module_id}/vae_drivers.parquet`.

## Captum Integrated Gradients

Requires optional install: `pip install isograph[torch-explain]`.

Captum Integrated Gradients (IG) attributes module eigengene prediction to individual
transcript features via the VAE encoder. Unlike decoder Jacobian attribution, IG captures
the full encoder nonlinearity and is complementary to the association-based approaches.

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table features.parquet \
  --feature-meta feature_metadata.parquet \
  --integrated-gradients \
  --ig-n-steps 100 \
  --ig-baseline zero \
  --output-dir artifacts/explain/run1
```

Output: `{module_id}/ig_attributions.parquet` — `feature_id`, `ig_score`, sorted by `|ig_score|`.

IG baseline options:

- `zero` (default) — attributes relative to no isoform switching
- `mean` — attributes relative to the per-sample gene mean usage
