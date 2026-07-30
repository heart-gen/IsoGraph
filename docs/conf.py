"""Sphinx configuration for IsoGraph documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "IsoGraph"
author = "Kynon J Benjamin"
copyright = "2026, Kynon J Benjamin"

try:
    from isograph import __version__ as release
except Exception:  # pragma: no cover - fallback if the package is not importable
    release = "0.1.4"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "staged-roadmap.md"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

autosummary_generate = False
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "description"

myst_enable_extensions = [
    "colon_fence",
]

# Generate implicit anchors for h1-h3 so cross-file links like
# `[...](configuration.md#residualization-is-a-discovery-only-knob)` resolve.
myst_heading_anchors = 3
