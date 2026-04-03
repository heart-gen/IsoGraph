from __future__ import annotations

from pathlib import Path

import pytest

from isograph.benchmarks.synthetic import generate_core_suite


@pytest.fixture()
def synthetic_suite(tmp_path: Path) -> Path:
    paths = generate_core_suite(tmp_path / "benchmarks" / "datasets", seed=7)
    return paths[0].parent
