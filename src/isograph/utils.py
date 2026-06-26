"""Shared utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataclass_to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return dataclass_to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: dataclass_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [dataclass_to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def stable_sign(scores: np.ndarray, loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orient a latent axis deterministically.

    The orientation pivot is the largest-magnitude loading. When several loadings
    are tied at (or within floating-point tolerance of) that magnitude -- which
    happens generically for symmetric axes such as the CLR PC1 of a two-transcript
    gene, whose loadings are ``[+a, -a]`` -- a bare ``argmax`` resolves the tie by
    raw index and is sensitive to sub-ULP noise. That noise depends on sample
    ordering, so the chosen pivot (and hence the axis sign) could flip purely from
    permuting samples, breaking permutation invariance downstream. Selecting the
    smallest index among the tolerance-tied maxima makes the orientation invariant
    for those symmetric cases while leaving a clear single maximum unchanged.
    """
    abs_loadings = np.abs(loadings)
    if abs_loadings.size == 0:
        return scores, loadings
    max_abs = abs_loadings.max()
    tied = np.flatnonzero(abs_loadings >= max_abs - 1e-9 * max_abs)
    pivot = int(tied[0])
    sign = 1.0 if loadings[pivot] >= 0 else -1.0
    return scores * sign, loadings * sign
