# Overview

IsoGraph is a research software package for gene-aware network analysis
of transcript-level RNA-seq data. The project is organized around reproducible datasets,
typed configurations, stable command-line entry points, and model backends that can be
compared on the same fixture suite.

## Documentation Map

- Use the `README.md` for project status, installation, and fast orientation.
- Use these reference docs for exact command behavior, data requirements, configuration
  fields, outputs, and Python APIs.
- Use the GitHub Wiki for tutorial-style walkthroughs, especially when preparing and
  analyzing your own data.

## Current Scope

IsoGraph currently includes:

- A deterministic **baseline** backend.
- A **latent** probabilistic backend (sklearn Factor Analysis + LedoitWolf partial correlation)
  with cross-validated component selection and stability selection support.
- A **graph-aware** backend extending the latent model with graph-Laplacian smoothing.
- A **VAE** backend — the default production backend — with nonlinear latent representation,
  early stopping, posterior-collapse diagnostics, and optional checkpointing. Requires PyTorch.
- A **WGCNA** backend wrapping R's `blockwiseModules` for direct comparison with WGCNA,
  including blockwise mode for datasets above 5 000 genes.
- Synthetic fixture suites: `core_v1` (24–800 genes), `scale_v1` (6 000–12 000 genes),
  and `multiplex_v1` — a typed abundance + switch suite with planted ground-truth roles
  (`switch_only`, `abundance_only`, `coupled`, `discordant`) across four fixtures
  (40–900 genes) plus an optional `xxlarge_multiplex_v1` at 12 000 genes.
- A real-data fixture freeze workflow for BrainSeq-style bulk RNA-seq inputs.
- **Multiplex network inference** (Stage 9A/9B) — each gene contributes a log-CPM
  abundance channel; multi-isoform genes also contribute a CLR-SVD switch channel.
  A typed feature graph with per-channel edge thresholds and auto-calibrated abundance
  threshold (`alpha_abundance_grid`) prevents spurious merging of switch modules.
  Gene channel role classification (`coupled`, `abundance_only`, `switch_only`,
  `discordant`) is reported in `module_gene_roles.parquet` for every fit.
- A **module explanation** module (`isograph.explain`) with `isograph explain-module` and
  `isograph annotate-structure` CLI subcommands for transcript-feature-level driver tables,
  publication-ready plots, VAE decoder attribution (Stage 8D), Captum Integrated Gradients
  encoder attribution (Stage 8E), and GTF-based structural annotation of switch pairs.
  Attribution outputs carry `feature_type` metadata so switch and abundance drivers are
  distinguished in multiplex fits.
- Multiplex stress reports for VAE, graph, latent, and WGCNA backends, including
  role-aware recall and giant-component diagnostics.
