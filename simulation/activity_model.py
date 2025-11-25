"""Activity inference heuristics for the virtual patients."""

from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np

from data_models import ActivityMode


def _circadian_activity_drive(hour: float) -> float:
    """Bimodal circadian activity drive with morning and afternoon peaks.

    Models typical human activity patterns:
    - Sleep: 23:00 - 06:00 (low activity drive ~0.0-0.1)
    - Morning wake/commute: 06:00 - 09:00 (moderate activity ~0.4-0.6)
    - Mid-morning work: 09:00 - 12:00 (moderate-low ~0.3-0.4)
    - Lunch break: 12:00 - 13:00 (low-moderate ~0.2-0.3)
    - Afternoon work: 13:00 - 17:00 (moderate ~0.3-0.5)
    - Evening activity: 17:00 - 20:00 (peak exercise window ~0.5-0.7)
    - Wind-down: 20:00 - 23:00 (declining ~0.2-0.1)
    """
    # Morning activity peak around 08:00
    morning_peak = 8.0
    morning_sigma = 2.0
    morning_drive = 0.5 * np.exp(-((hour - morning_peak) ** 2) / (2 * morning_sigma ** 2))

    # Afternoon/evening activity peak around 18:00 (post-work exercise window)
    evening_peak = 18.0
    evening_sigma = 2.5
    evening_drive = 0.6 * np.exp(-((hour - evening_peak) ** 2) / (2 * evening_sigma ** 2))

    # Sleep suppression: strong suppression from 23:00 to 06:00
    if hour < 6.0 or hour >= 23.0:
        sleep_factor = 0.05
    elif hour < 7.0:
        # Gradual wake-up from 6-7 AM
        sleep_factor = 0.05 + (hour - 6.0) * 0.45
    elif hour >= 22.0:
        # Gradual wind-down from 22-23
        sleep_factor = 0.5 - (hour - 22.0) * 0.45
    else:
        sleep_factor = 0.5

    # Combine drives with sleep modulation
    combined = (morning_drive + evening_drive) * (0.5 + sleep_factor)
    return float(np.clip(combined, 0.0, 1.0))


def determine_activity(
    timestamp: datetime,
    rng: np.random.Generator,
) -> Tuple[ActivityMode, float]:
    """
    Infer activity mode and intensity using time of day with stochastic variation.

    The bimodal circadian drive produces higher activation near morning/evening,
    with realistic random variation for individual behavior differences.
    """

    hour = timestamp.hour + timestamp.minute / 60.0
    circadian = _circadian_activity_drive(hour)

    # Stochastic component: individual variation in activity level
    # Lower mean (0.1) to prevent excessive activity classification
    noise = float(np.clip(rng.normal(0.1, 0.15), -0.1, 0.4))
    base_intensity = float(np.clip(circadian + noise, 0.0, 1.0))

    # Classify activity mode based on intensity
    # Thresholds designed for realistic activity distribution:
    # - ~35% sleep/sedentary, ~40% light activity, ~20% moderate, ~5% intense
    if base_intensity < 0.15:
        # Low intensity: sleep during night hours, sedentary otherwise
        if hour < 6.0 or hour >= 22.5:
            mode = ActivityMode.SLEEP
        else:
            mode = ActivityMode.SEDENTARY
    elif base_intensity < 0.35:
        mode = ActivityMode.SEDENTARY
    elif base_intensity < 0.55:
        mode = ActivityMode.WALKING
    elif base_intensity < 0.80:
        mode = ActivityMode.RUNNING
    else:
        # Rare stress/high-intensity events
        mode = ActivityMode.STRESS_EVENT

    return mode, base_intensity

