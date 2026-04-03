from __future__ import annotations

from isograph.io.artifacts import load_dataset_bundle


def test_core_suite_generation(synthetic_suite) -> None:
    toy = load_dataset_bundle(synthetic_suite / "toy_v1")
    medium = load_dataset_bundle(synthetic_suite / "medium_v1")
    assert toy.manifest.dataset_name == "toy_v1"
    assert medium.manifest.dataset_name == "medium_v1"
    assert toy.matrices["transcript_counts"].shape[1] == 48
    assert medium.matrices["transcript_counts"].shape[1] == 240
