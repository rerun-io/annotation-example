from __future__ import annotations

from typing import SupportsFloat

import numpy as np
from jaxtyping import Float32
from numpy import ndarray


FilterArray = Float32[ndarray, "..."]
ScalarOrArray = float | FilterArray


def smoothing_factor(t_e: float, cutoff: ScalarOrArray) -> ScalarOrArray:
    """Return exponential smoothing factor for a time step and cutoff frequency.

    Args:
        t_e: Elapsed time since the previous sample.
        cutoff: Low-pass filter cutoff frequency; may broadcast to the signal shape.

    Returns:
        ScalarOrArray: The smoothing coefficient matching the shape of ``cutoff``.
    """

    r: ScalarOrArray = 2.0 * np.pi * cutoff * t_e
    result: ScalarOrArray = r / (r + 1.0)
    if isinstance(result, np.ndarray):
        return result.astype(np.float32, copy=False)
    return float(result)


def exponential_smoothing(a: ScalarOrArray, x: FilterArray, x_prev: FilterArray) -> FilterArray:
    """Blend current and previous samples with coefficient ``a``.

    Args:
        a: Smoothing coefficient in ``[0, 1]`` matching the broadcast shape of ``x``.
        x: Current observation.
        x_prev: Previous filtered observation.

    Returns:
        FilterArray: Smoothed sample following an exponential moving average.
    """

    a_array: FilterArray = np.asarray(a, dtype=np.float32)
    smoothed: FilterArray = (a_array * x) + ((1.0 - a_array) * x_prev)
    return smoothed


class OneEuroFilter:
    """Apply the One Euro filter to smooth noisy signals while preserving dynamics."""

    def __init__(
        self,
        t0: SupportsFloat,
        x0: FilterArray,
        dx0: FilterArray | SupportsFloat = 0.0,
        min_cutoff: SupportsFloat = 1.0,
        beta: SupportsFloat = 0.0,
        d_cutoff: SupportsFloat = 1.0,
    ) -> None:
        """Create a filter configured for the provided initial sample.

        Args:
            t0: Timestamp corresponding to the initial observation ``x0``.
            x0: Initial observation that seeds the filter history.
            dx0: Initial estimate of the derivative; defaults to zeros matching ``x0``.
            min_cutoff: Baseline cutoff frequency controlling smoothing strength.
            beta: Speed coefficient increasing the cutoff when motion is high.
            d_cutoff: Cutoff frequency applied to the derivative term.

        Raises:
            ValueError: If ``dx0`` is provided as an array with a shape different from ``x0``.
        """

        x0_array: FilterArray = np.asarray(x0, dtype=np.float32)
        if np.isscalar(dx0):
            dx0_array: FilterArray = np.zeros_like(x0_array)
        else:
            dx0_array = np.asarray(dx0, dtype=np.float32)
            if dx0_array.shape != x0_array.shape:
                msg = "`dx0` must match the shape of `x0` when provided as an array"
                raise ValueError(msg)

        self.min_cutoff: float = float(min_cutoff)
        self.beta: float = float(beta)
        self.d_cutoff: float = float(d_cutoff)
        self.x_prev: FilterArray = x0_array
        self.dx_prev: FilterArray = dx0_array
        self.t_prev: float = float(t0)

    def __call__(self, t: SupportsFloat, x: FilterArray) -> FilterArray:
        """Filter ``x`` observed at time ``t`` and return the smoothed sample.

        Args:
            t: Timestamp of the new observation. Must be strictly greater than the previous call.
            x: Observation to smooth; must match the shape used during initialization.

        Returns:
            FilterArray: Filtered observation with the same shape as ``x``.

        Raises:
            ValueError: If timestamps are non-increasing or the shape of ``x`` differs from ``x0``.
        """

        t_float: float = float(t)
        t_e: float = t_float - self.t_prev
        if t_e <= 0.0:
            msg = "timestamps must be strictly increasing"
            raise ValueError(msg)

        x_array: FilterArray = np.asarray(x, dtype=np.float32)
        if x_array.shape != self.x_prev.shape:
            msg = "incoming sample shape must match the initialization shape"
            raise ValueError(msg)

        a_d: float = float(smoothing_factor(t_e, self.d_cutoff))
        dx: FilterArray = (x_array - self.x_prev) / np.float32(t_e)
        dx_hat: FilterArray = exponential_smoothing(a_d, dx, self.dx_prev)

        dx_abs: FilterArray = np.abs(dx_hat)
        cutoff: ScalarOrArray = self.min_cutoff + (self.beta * dx_abs)
        a: ScalarOrArray = smoothing_factor(t_e, cutoff)
        x_hat: FilterArray = exponential_smoothing(a, x_array, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t_float

        return x_hat
