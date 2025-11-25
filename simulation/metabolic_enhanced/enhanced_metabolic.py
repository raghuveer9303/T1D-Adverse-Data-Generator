"""Enhanced metabolic model with improved insulin pharmacokinetics.

This module provides a drop-in replacement for calculate_next_glucose()
with improved clinical accuracy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from simulation.metabolic_enhanced.insulin_pools import (
    InsulinPools,
    evolve_insulin_pools,
)
from simulation.metabolic_enhanced.insulin_action import (
    InsulinActionBuffer,
    calculate_insulin_action,
)
from simulation.metabolic_enhanced.dawn_phenomenon import (
    calculate_dawn_glucose_rise,
    calculate_dawn_insulin_resistance,
)
from simulation.metabolic_enhanced.bolus_calculator import (
    calculate_bolus_insulin,
)

# Import constants from original model for compatibility
from simulation.metabolic_model import (
    GLUCOSE_MIN,
    GLUCOSE_MAX,
    HEPATIC_GLUCOSE_TARGET,
)


def calculate_next_glucose_enhanced(
    current_glucose: float,
    insulin_pools: InsulinPools,
    insulin_action_buffer: InsulinActionBuffer,
    carbs_in_stomach: float,
    sensitivity_factor: float,
    activity_intensity: float,
    hour_of_day: float,
    new_bolus: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, InsulinPools, float]:
    """
    Enhanced glucose evolution with improved insulin modeling.
    
    Models:
    - Separate rapid-acting and basal insulin pools
    - Insulin action delays (15-30 min delay before peak)
    - Dawn phenomenon (4-8 AM glucose rise and insulin resistance)
    - Carbohydrate absorption
    - Exercise-induced glucose utilization
    - Hepatic glucose homeostasis
    
    Args:
        current_glucose: Current blood glucose (mg/dL)
        insulin_pools: Separate rapid-acting and basal insulin pools
        insulin_action_buffer: Buffer tracking insulin action delays
        carbs_in_stomach: Carbohydrates being digested (grams)
        sensitivity_factor: Patient's base insulin sensitivity
        activity_intensity: Current activity level (0.0-1.0)
        hour_of_day: Hour of day (0-24, including minutes)
        rng: Random number generator
    
    Returns:
        Tuple of (new_glucose, updated_insulin_pools, updated_carbs_in_stomach)
    """
    rng = rng or np.random.default_rng()

    glucose = current_glucose

    # =====================================================================
    # 1. DAWN PHENOMENON (4-8 AM)
    # =====================================================================
    dawn_glucose_rise = calculate_dawn_glucose_rise(hour_of_day)
    glucose += dawn_glucose_rise

    dawn_resistance = calculate_dawn_insulin_resistance(hour_of_day)
    effective_sensitivity = sensitivity_factor / dawn_resistance

    # =====================================================================
    # 2. CARBOHYDRATE ABSORPTION
    # =====================================================================
    # Rate varies: slower at high carb loads, faster when nearly empty
    absorption_rate = 0.5 if carbs_in_stomach > 10.0 else 0.3
    carbs_absorbed = min(carbs_in_stomach, absorption_rate)
    glucose += carbs_absorbed * 8.0  # ~8 mg/dL rise per gram absorbed
    carbs_in_stomach = max(0.0, carbs_in_stomach - carbs_absorbed)

    # =====================================================================
    # 3. INSULIN ACTION (with delays)
    # =====================================================================
    insulin_activity = calculate_insulin_action(
        buffer=insulin_action_buffer,
        insulin_pools=insulin_pools,
        sensitivity_factor=effective_sensitivity,
        new_bolus=new_bolus,
    )
    glucose -= insulin_activity

    # =====================================================================
    # 4. EVOLVE INSULIN POOLS
    # =====================================================================
    insulin_pools = evolve_insulin_pools(
        pools=insulin_pools,
        add_basal=True,
        add_bolus=0.0,  # No new bolus this minute
    )

    # =====================================================================
    # 5. EXERCISE-INDUCED GLUCOSE UTILIZATION
    # =====================================================================
    exercise_uptake = activity_intensity * 3.0
    glucose -= exercise_uptake

    # =====================================================================
    # 6. HEPATIC GLUCOSE HOMEOSTASIS
    # =====================================================================
    if glucose < HEPATIC_GLUCOSE_TARGET:
        hepatic_release = (HEPATIC_GLUCOSE_TARGET - glucose) * 0.02
        glucose += hepatic_release

    # =====================================================================
    # 7. STOCHASTIC PHYSIOLOGICAL NOISE
    # =====================================================================
    glucose += rng.normal(0.0, 1.5)

    # =====================================================================
    # 8. CLAMP TO PHYSIOLOGICAL BOUNDS
    # =====================================================================
    glucose = float(np.clip(glucose, GLUCOSE_MIN, GLUCOSE_MAX))
    carbs_in_stomach = max(0.0, float(carbs_in_stomach))

    return glucose, insulin_pools, carbs_in_stomach

