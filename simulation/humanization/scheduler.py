"""Stochastic schedulers for meals and weekend circadian shifts."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np

MEAL_WINDOWS = {
    "breakfast": (7, 9),
    "lunch": (12, 14),
    "dinner": (18, 20),
}

MEAL_CARBS = {
    "breakfast": 45.0,
    "lunch": 60.0,
    "dinner": 65.0,
}

MEAL_PROBABILITY_PER_MINUTE = 0.005  # 0.5% chance each minute inside window.


def adjust_for_weekend_wake_shift(timestamp: datetime) -> datetime:
    """Shift the wake drive forward on weekends to emulate sleeping in."""

    if timestamp.weekday() >= 5:
        return timestamp + timedelta(hours=2)
    return timestamp


def maybe_trigger_meal_event(
    timestamp: datetime,
    rng: np.random.Generator,
    daily_meal_flags: Dict[str, bool],
    last_meal_time: Optional[datetime],
) -> Tuple[Optional[Tuple[str, float]], Dict[str, bool]]:
    """
    Sample stochastic meals during their allowed windows.

    Ensures we only emit a lunch event once per day even if flags drift.
    """

    hour = timestamp.hour + timestamp.minute / 60.0
    updated_flags = dict(daily_meal_flags)
    rng = rng or np.random.default_rng()

    for meal_name, (start_hour, end_hour) in MEAL_WINDOWS.items():
        if updated_flags.get(meal_name, False):
            continue
        if not (start_hour <= hour < end_hour):
            continue
        if meal_name == "lunch" and last_meal_time:
            same_day = last_meal_time.date() == timestamp.date()
            lunch_window = 12 <= last_meal_time.hour < 14
            if same_day and lunch_window:
                continue
        if rng.random() < MEAL_PROBABILITY_PER_MINUTE:
            updated_flags[meal_name] = True
            return (meal_name, MEAL_CARBS[meal_name]), updated_flags

    return None, updated_flags


