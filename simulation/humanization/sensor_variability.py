"""Activity dependent sensor noise helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np


def apply_sensor_noise(
    true_value: float,
    activity_intensity: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Apply Gaussian noise where variance increases with movement intensity.
    """

    rng = rng or np.random.default_rng()
    if activity_intensity < 0.1:
        sigma = 0.5
    elif activity_intensity > 0.5:
        sigma = 3.0
    else:
        sigma = 1.5
    return float(true_value + rng.normal(0.0, sigma))
