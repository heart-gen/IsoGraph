from __future__ import annotations

import pytest
from pydantic import ValidationError

from isograph.validation import DatasetManifest


def test_manifest_rejects_missing_feature_tables() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest(
            dataset_name="toy_v1",
            suite_name="core_v1",
            description="bad",
            sample_table="samples.parquet",
            feature_tables=[],
            matrices=[],
            provenance={},
        )
