# Overview

IsoGraph is a benchmark-first research software package for gene-aware network analysis
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
- A **GPU-latent** backend using the Woodbury identity for memory-efficient Factor Analysis
  (avoids forming the p×p covariance matrix) with BIC-based component selection. Requires PyTorch.
- Synthetic fixture suites: `core_v1` (24–800 genes) and `scale_v1` (6 000–12 000 genes).
- A real-data fixture freeze workflow for BrainSeq-style bulk RNA-seq inputs.

The development roadmap in `docs/staged-roadmap.md` records stage history and planned
next work.
