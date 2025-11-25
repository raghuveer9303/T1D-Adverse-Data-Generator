"""Metabolic transitions that evolve glucose and related stores."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# Physiological glucose bounds (mg/dL)
GLUCOSE_MIN = 40.0   # Severe hypoglycemia threshold
GLUCOSE_MAX = 600.0  # Extreme hyperglycemia (diabetic ketoacidosis territory)

# Basal insulin infusion rate (units/minute) - typical basal rate ~0.5-1.0 units/hour
BASAL_INSULIN_RATE = 0.015  # ~0.9 units/hour

# Homeostatic glucose target for liver glucose release/uptake
HEPATIC_GLUCOSE_TARGET = 100.0


def calculate_next_glucose(
    current_glucose: float,
    insulin_on_board: float,
    carbs_in_stomach: float,
    sensitivity_factor: float,
    activity_intensity: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Advance glucose by one minute using a simplified pharmacokinetic model.

    Models:
    - Carbohydrate absorption from gut (~0.5g/min max absorption rate)
    - Insulin-mediated glucose uptake (dependent on sensitivity factor)
    - Exercise-induced glucose utilization
    - Hepatic glucose homeostasis (liver releases glucose when low)
    - Basal insulin replenishment (continuous pump/injection simulation)

    Returns updated (glucose, insulin_on_board, carbs_in_stomach).
    """
    rng = rng or np.random.default_rng()

    glucose = current_glucose

    # Carbohydrate absorption from stomach (gastric emptying + intestinal absorption)
    # Rate varies: slower at high carb loads, faster when nearly empty
    absorption_rate = 0.5 if carbs_in_stomach > 10.0 else 0.3
    carbs_absorbed = min(carbs_in_stomach, absorption_rate)
    glucose += carbs_absorbed * 8.0  # ~8 mg/dL rise per gram absorbed
    carbs_in_stomach = max(0.0, carbs_in_stomach - carbs_absorbed)

    # Insulin-mediated glucose uptake
    # Higher sensitivity = more glucose cleared per unit insulin
    insulin_activity = insulin_on_board * sensitivity_factor * 2.5
    glucose -= insulin_activity

    # Insulin decay (pharmacokinetic half-life ~4-6 hours for rapid-acting)
    # 2% decay per minute gives ~35 min half-life (rapid-acting analog)
    insulin_on_board *= 0.98

    # Basal insulin replenishment (simulates pump basal rate or long-acting injection)
    insulin_on_board += BASAL_INSULIN_RATE

    # Exercise-induced glucose utilization (muscle uptake independent of insulin)
    exercise_uptake = activity_intensity * 3.0
    glucose -= exercise_uptake

    # Hepatic glucose homeostasis: liver releases glucose when blood sugar drops
    # This prevents glucose from crashing unrealistically during fasting
    if glucose < HEPATIC_GLUCOSE_TARGET:
        hepatic_release = (HEPATIC_GLUCOSE_TARGET - glucose) * 0.02
        glucose += hepatic_release

    # Stochastic physiological noise (meal timing, stress hormones, etc.)
    glucose += rng.normal(0.0, 1.5)

    # Clamp to physiological bounds
    glucose = float(np.clip(glucose, GLUCOSE_MIN, GLUCOSE_MAX))
    insulin_on_board = max(0.0, float(insulin_on_board))
    carbs_in_stomach = max(0.0, float(carbs_in_stomach))

    return glucose, insulin_on_board, carbs_in_stomach

