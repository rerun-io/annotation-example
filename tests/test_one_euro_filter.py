from __future__ import annotations

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from annotation_example.one_euro_filter import OneEuroFilter, smoothing_factor


def test_one_euro_filter_first_update_matches_expected() -> None:
    x0: Float32[ndarray, "2"] = np.zeros(2, dtype=np.float32)
    filter_1e: OneEuroFilter = OneEuroFilter(
        t0=0.0,
        x0=x0,
        min_cutoff=1.0,
        beta=0.0,
        d_cutoff=1.0,
    )

    new_sample: Float32[ndarray, "2"] = np.array([1.0, -1.0], dtype=np.float32)
    filtered: Float32[ndarray, "2"] = filter_1e(1.0, new_sample)

    alpha: float = smoothing_factor(1.0, 1.0)
    expected: Float32[ndarray, "2"] = (alpha * new_sample) + ((1.0 - alpha) * x0)

    assert np.allclose(filtered, expected)
