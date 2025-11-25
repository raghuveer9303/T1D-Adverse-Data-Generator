"""Circadian rhythm helpers that add human-like drift to physiology."""

from __future__ import annotations

import numpy as np


def apply_circadian_drift(
    baseline_value: float,
    hour_of_day: float,
    amplitude: float,
) -> float:
    """
    Apply a 24 hour sinusoidal drift with a trough near 03:00 and peak at 15:00.

    Args:
        baseline_value: The baseline magnitude around which we oscillate.
        hour_of_day: Hour in local time as float (0-24) including minutes.
        amplitude: +/- deviation applied at peak/trough.
    """

    normalized_hour = hour_of_day / 24.0
    phase_shift = -0.75 * np.pi  # Align minimum to ~03:00 and max to ~15:00.
    angle = 2.0 * np.pi * normalized_hour + phase_shift
    return float(baseline_value + amplitude * np.sin(angle))


