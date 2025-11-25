"""Dawn phenomenon modeling.

The dawn phenomenon is a natural rise in blood glucose between 4-8 AM
due to growth hormone and cortisol release, causing temporary insulin resistance.
"""

from __future__ import annotations

from typing import Optional


def calculate_dawn_glucose_rise(hour_of_day: float) -> float:
    """
    Calculate glucose rise due to dawn phenomenon.
    
    Dawn phenomenon occurs between 4-8 AM with peak around 6 AM.
    Can cause 20-50 mg/dL rise in glucose.
    
    Args:
        hour_of_day: Hour of day (0-24, including minutes as fraction)
    
    Returns:
        Glucose rise in mg/dL (0-30 mg/dL typical)
    """
    # Dawn phenomenon window: 4-8 AM
    if hour_of_day < 4.0 or hour_of_day >= 8.0:
        return 0.0

    # Peak effect around 6 AM
    peak_hour = 6.0
    distance_from_peak = abs(hour_of_day - peak_hour)

    # Maximum rise of 30 mg/dL at peak
    max_rise = 30.0

    # Linear decrease from peak (simplified model)
    # At 4 AM and 8 AM: 0 mg/dL rise
    # At 6 AM: 30 mg/dL rise
    if distance_from_peak <= 2.0:
        # Within 2 hours of peak
        severity = 1.0 - (distance_from_peak / 2.0)
        return max_rise * severity
    else:
        return 0.0


def calculate_dawn_insulin_resistance(hour_of_day: float) -> float:
    """
    Calculate insulin resistance multiplier due to dawn phenomenon.
    
    Insulin resistance increases during dawn phenomenon, reducing
    the effectiveness of insulin by 20-30%.
    
    Args:
        hour_of_day: Hour of day (0-24, including minutes as fraction)
    
    Returns:
        Resistance multiplier (1.0 = no resistance, 1.3 = 30% resistance)
    """
    # Dawn phenomenon window: 4-8 AM
    if hour_of_day < 4.0 or hour_of_day >= 8.0:
        return 1.0

    # Peak resistance around 6 AM
    peak_hour = 6.0
    distance_from_peak = abs(hour_of_day - peak_hour)

    # Maximum resistance: 30% (multiplier of 1.3)
    max_resistance = 0.3

    # Linear decrease from peak
    if distance_from_peak <= 2.0:
        severity = 1.0 - (distance_from_peak / 2.0)
        resistance = max_resistance * severity
        return 1.0 + resistance
    else:
        return 1.0

