"""Bolus insulin calculation for meals and corrections.

Calculates appropriate bolus insulin doses based on:
- Carbohydrate intake (carb ratio)
- Current glucose level (correction factor)
- Target glucose range
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Default insulin-to-carb ratio: 1 unit per 15g carbs
# This varies by patient (typically 1:10 to 1:20)
DEFAULT_IC_RATIO = 1.0 / 15.0

# Default correction factor: 1 unit lowers glucose by 50 mg/dL
# This varies by patient (typically 30-100 mg/dL per unit)
DEFAULT_CORRECTION_FACTOR = 50.0

# Target glucose range
TARGET_GLUCOSE_LOW = 80.0
TARGET_GLUCOSE_HIGH = 120.0
TARGET_GLUCOSE_MID = 100.0


def calculate_bolus_insulin(
    carbs_grams: float,
    current_glucose: float,
    ic_ratio: Optional[float] = None,
    correction_factor: Optional[float] = None,
    target_glucose: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Calculate bolus insulin dose for a meal.
    
    Bolus = (Carbs × IC Ratio) + Correction
    
    Where Correction = (Current Glucose - Target) / Correction Factor
    
    Args:
        carbs_grams: Carbohydrate content of meal (grams)
        current_glucose: Current blood glucose (mg/dL)
        ic_ratio: Insulin-to-carb ratio (units per gram). Default: 1/15
        correction_factor: How much 1 unit lowers glucose (mg/dL). Default: 50
        target_glucose: Target glucose level (mg/dL). Default: 100
        rng: Random generator for patient-specific variation
    
    Returns:
        Bolus insulin dose in units
    """
    rng = rng or np.random.default_rng()

    # Use defaults if not provided
    ic_ratio = ic_ratio or DEFAULT_IC_RATIO
    correction_factor = correction_factor or DEFAULT_CORRECTION_FACTOR
    target_glucose = target_glucose or TARGET_GLUCOSE_MID

    # Carb coverage: insulin needed for meal
    carb_bolus = carbs_grams * ic_ratio

    # Correction: insulin needed to bring glucose to target
    glucose_above_target = max(0.0, current_glucose - target_glucose)
    correction_bolus = glucose_above_target / correction_factor

    # Total bolus
    total_bolus = carb_bolus + correction_bolus

    # Add small random variation (±5%) to model patient behavior
    # (people don't calculate perfectly)
    variation = rng.normal(1.0, 0.05)
    total_bolus *= variation

    # Round to nearest 0.5 units (typical pump precision)
    total_bolus = round(total_bolus * 2.0) / 2.0

    # Ensure non-negative
    return max(0.0, float(total_bolus))

