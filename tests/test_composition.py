from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from isograph.features.composition import clr_transform, inverse_clr, inverse_logit, logit_transform


@given(
    st.lists(
        st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=2, max_size=4),
        min_size=2,
        max_size=4,
    )
)
def test_clr_inverse_returns_valid_composition(rows: list[list[float]]) -> None:
    width = len(rows[0])
    assume_same_width = all(len(row) == width for row in rows)
    if not assume_same_width:
        return
    matrix = np.array(rows, dtype=float)
    matrix = matrix / matrix.sum(axis=0, keepdims=True)
    restored = inverse_clr(clr_transform(matrix))
    assert np.allclose(restored.sum(axis=0), 1.0)
    assert np.all(restored >= 0.0)


@given(st.lists(st.floats(min_value=0.001, max_value=0.999), min_size=4, max_size=12))
def test_logit_inverse_round_trip(values: list[float]) -> None:
    arr = np.array(values, dtype=float)[None, :]
    restored = inverse_logit(logit_transform(arr))
    assert np.allclose(arr, restored)
