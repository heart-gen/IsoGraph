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

- A deterministic baseline backend.
- A latent probabilistic backend with stability selection support.
- A graph-aware backend.
- A VAE backend with an optional PyTorch dependency.
- Synthetic and real-data fixture workflows centered on the `core_v1` suite.

The development roadmap in `docs/staged-roadmap.md` records stage history and planned
next work.
