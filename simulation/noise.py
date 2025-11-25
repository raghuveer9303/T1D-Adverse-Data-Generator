"""Noise helpers applied uniformly across sensor modalities."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def apply_sensor_noise(
    value: Union[int, float],
    sigma: float,
    rng: Optional[np.random.Generator] = None,
    signal_loss_probability: float = 0.01,
) -> Optional[float]:
    """Inject Gaussian perturbations and occasional signal dropouts."""

    rng = rng or np.random.default_rng()
    if rng.random() < signal_loss_probability:
        return None
    noisy = float(value) + float(rng.normal(0.0, sigma))
    return noisy

