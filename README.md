# IsoGraph

IsoGraph discovers co-regulated transcript programs from bulk RNA-seq by treating gene
abundance and isoform switching as separate channels in a multiplex network. Starting
from transcript-level counts, it builds gene-local switch coordinates from compositional
transcript usage and a standardized abundance signal per gene, infers sparse gene-module
structure via a VAE or linear backend, classifies each module gene by the channel driving
its membership (switching, abundance, coupled, or discordant), and links modules to
phenotypic traits. All steps are benchmark-validated and reproducible.

## Core Capabilities

- **Multiplex network inference** — builds a typed feature graph with separate abundance
  and switch channels per gene, calibrated edge thresholds, and gene channel role
  classification (`coupled`, `switch_only`, `abundance_only`, `discordant`).
- **Five interchangeable backends** — `baseline`, `latent`, `graph`, `vae` (default),
  and `wgcna`; all support multiplex mode and the full benchmark suite.
- **Benchmark validation** — three fixture suites: `core_v1` (24–800 genes),
  `scale_v1` (6k–12k genes), and `multiplex_v1` (typed abundance + switch ground truth).
  An optional `xxlarge_multiplex_v1` fixture (12k genes, 240 samples) tests at full scale.
- **Module explanation** — `isograph explain-module` produces gene driver tables,
  transcript polarity tables, high-vs-low contrasts, publication-ready PDF plots, optional
  VAE decoder attribution, and Captum Integrated Gradients encoder attribution.
- **Structural annotation** — `isograph annotate-structure` labels switch pairs with
  GTF-derived exon, CDS/UTR, biotype, and coding-status changes.
- **Reproducibility** — fixture-driven, seed-controlled, version-locked YAML configs;
  compact JSON reports as durable evidence; real-data freeze via `freeze-real`.

## Installation

Install the core package from PyPI:

```bash
pip install isograph
```

The core package supports Python `3.11` through `3.14`.

### Optional backends

IsoGraph installs `mpmath`, which is required by modern SymPy releases. The
`vae` backend also requires PyTorch, but PyTorch is intentionally not installed
by IsoGraph because CPU/GPU/CUDA builds are platform-specific. Install the build
that matches your system before using it:

```bash
pip install torch
```

See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for
GPU/CUDA builds.

The `wgcna` backend requires R with the `WGCNA` package and `Rscript` on `PATH`.

## Quickstart

### Step 1 — Validate on bundled fixtures

Run the multiplex benchmark on the toy fixture to confirm your installation (VAE default,
~2 minutes):

```bash
isograph benchmark --config-name stage9_multiplex_vae \
  -- fixture_filter=toy_multiplex_v1 stage_name=readme_smoke
```

This writes artifacts under `artifacts/benchmarks/readme_smoke/toy_multiplex_v1/` and
a JSON report under `artifacts/reports/`, including overall recovery and role-aware
recall (`switch_only`, `abundance_only`, `coupled`, `discordant`).

Run all four backends on the full multiplex suite:

```bash
isograph benchmark --config-name stage9_multiplex_vae
isograph benchmark --config-name stage9_multiplex_graph
isograph benchmark --config-name stage9_multiplex_latent
isograph benchmark --config-name stage9_multiplex_wgcna
```

The optional 12k-gene stress fixture:

```bash
isograph benchmark --config-name stress_multiplex_xxlarge_vae
isograph benchmark --config-name stress_multiplex_xxlarge_wgcna
```

### Step 2 — Fit your own data

IsoGraph expects a prepared dataset bundle with a `manifest.json`, sample metadata,
feature tables, and count matrices. Abundance channels are always computed (from
`gene_counts` if provided, otherwise summed from `transcript_counts`) and appear in
`feature_scores` for trait associations and role classification. Linear backends
(baseline, latent, graph) use switch-only channels for graph edge inference by default;
set `allow_abundance_abundance=True` or `alpha_abundance_grid=[...]` to enable
abundance-abundance edges (multiplex mode).

```bash
isograph fit \
  --dataset-path path/to/my_dataset_bundle \
  --output-dir artifacts/fits/my_dataset
```

With tuning overrides (all after `--`):

```bash
isograph fit \
  --dataset-path path/to/my_dataset_bundle \
  --backend vae \
  --output-dir artifacts/fits/my_dataset_vae \
  -- vae.hidden_dim=256 vae.latent_dim=8 vae.n_epochs=500 \
     vae.alpha=0.70 vae.alpha_switch=0.70 \
     vae.alpha_abundance_grid="[0.60,0.65,0.70,0.75,0.80,0.85,0.90]"
```

**What you get:**

| File | Contents |
|---|---|
| `modules.parquet` | Module assignments per gene |
| `edges.parquet` | Inferred gene-gene edges with weights |
| `traits.parquet` | Module–trait associations and p values |
| `feature_scores.parquet` | Per-gene switch and abundance scores with `feature_type` column |
| `module_gene_roles.parquet` | Per-gene role: `coupled`, `switch_only`, `abundance_only`, `discordant` |

### Step 3 — Explain modules

```bash
isograph explain-module \
  --artifact-dir artifacts/fits/my_dataset \
  --feature-table path/to/feature_scores.parquet \
  --feature-meta path/to/transcript_table.parquet \
  --module-ids M000 M001 \
  --plot --output-format pdf \
  --output-dir artifacts/explain/my_dataset
```

Produces gene driver tables, transcript polarity tables, high-vs-low contrasts, and
publication-ready PDF plots per module.

The detailed walkthroughs live in the Wiki, and the formal data model is documented
in the RTD source tree.

## Documentation

- Details on all of the functions lives in the Read the Docs 
  [API](https://isograph.readthedocs.io/en/latest/).
- Step-by-step tutorials for installation, data preparation, and own-data workflows live
  in the [GitHub Wiki](https://github.com/heart-gen/IsoGraph/wiki).

## Citation

If you use IsoGraph in research, cite the software repository using the metadata in
[CITATION.cff](CITATION.cff). If a manuscript or preprint becomes available later, that
can be added as a preferred citation target without changing the software citation path.

## Acknowledgements

IsoGraph is supported by the National Institute on Minority Health and Health Disparities
award `R00 MD0169640` and the Alzheimer's Association award `25AARG-1413315`.

## Reproducibility and Data Provenance

- The benchmark suite is fixture-driven and designed to preserve regression targets across
  development stages.
- The bundled real-data workflow freezes a reproducible `real_caudate_aa_v1` dataset from
  BrainSEQ-derived inputs and caches intermediate selections under
  `benchmarks/cache/real_data/`.
- Benchmark, calibration, runtime, and snapshot artifacts are written into versioned
  directories under `artifacts/` and `snapshots/`.
- Bulky generated benchmark directories and dataset bundles are ignored by git. Commit
  durable benchmark evidence as compact JSON reports under `artifacts/reports/`.

## Limitations

- The benchmark CLI is optimized for the bundled fixture suite rather than arbitrary
  user-defined suites.
- The VAE backend requires a separate PyTorch installation.
- The WGCNA backend requires R with the `WGCNA` package installed.
- The `freeze-real` workflow depends on BrainSEQ-style source files and is not a
  generic data-ingestion command for arbitrary cohorts.
- VAE decoder attribution (`--vae-attribution`) and Captum Integrated Gradients
  (`--integrated-gradients`) require a VAE checkpoint in the fit artifact directory and,
  for Integrated Gradients, `pip install isograph[torch-explain]`.
