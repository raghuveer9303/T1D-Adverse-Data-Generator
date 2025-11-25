"""Core minute-wise evolution logic for the diabetic digital twin."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Iterable, List, Optional, Tuple
import uuid

import numpy as np

from data_models import (
    ActivityMode,
    PatientDynamicState,
    PatientStaticProfile,
    SensorPayload,
)
from simulation.activity_model import determine_activity
from simulation.metabolic_model import calculate_next_glucose
from simulation.noise import apply_sensor_noise as apply_generic_sensor_noise
from simulation.signal_model import (
    calculate_vitals_target,
    generate_waveform_snapshot,
)
from simulation.humanization.circadian import apply_circadian_drift
from simulation.humanization.scheduler import (
    adjust_for_weekend_wake_shift,
    maybe_trigger_meal_event,
)
from simulation.humanization.sensor_variability import apply_sensor_noise
# Enhanced metabolic modeling
from simulation.metabolic_enhanced.insulin_pools import (
    InsulinPools,
    evolve_insulin_pools,
)
from simulation.metabolic_enhanced.insulin_action import InsulinActionBuffer
from simulation.metabolic_enhanced.enhanced_metabolic import calculate_next_glucose_enhanced
from simulation.metabolic_enhanced.bolus_calculator import calculate_bolus_insulin


def _update_fatigue(current: float, intensity: float) -> float:
    """Fatigue rises with intensity and decays slowly at rest."""
    recovered = current * 0.97
    load = intensity * 0.05
    return float(np.clip(recovered + load, 0.0, 1.0))


def _trend_arrow(delta: float) -> str:
    """Translate glucose slope into CGM trend jargon."""
    if delta > 5.0:
        return "RISING_QUICKLY"
    if delta > 2.0:
        return "RISING"
    if delta < -5.0:
        return "FALLING_QUICKLY"
    if delta < -2.0:
        return "FALLING"
    return "STABLE"


def _default_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng or np.random.default_rng()


def _fresh_daily_meal_flags() -> dict:
    return {"breakfast": False, "lunch": False, "dinner": False}


def process_single_patient(
    profile: PatientStaticProfile,
    state: PatientDynamicState,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[PatientDynamicState, SensorPayload]:
    """
    Propagate a single patient forward by one minute.

    1. Infer activity via circadian Gaussian wake drive.
    2. Move metabolic buffers and glucose using pharmacodynamic heuristics.
    3. Convert metabolic stress into vitals (sympathetic responses).
    4. Render ECG/EDA snippets with NeuroKit2 (10-second window).
    5. Assemble the SensorPayload for downstream Pub/Sub streaming.
    """

    rng = _default_rng(rng)
    current_glucose = state.metabolic_state.get("glucose_true_mgdl", 110.0)
    insulin_on_board = state.metabolic_state.get("insulin_on_board_units", 0.5)
    carbs_in_stomach = state.metabolic_state.get("carbs_in_stomach_grams", 0.0)
    hour_of_day = state.timestamp_utc.hour + state.timestamp_utc.minute / 60.0
    activity_timestamp = adjust_for_weekend_wake_shift(state.timestamp_utc)

    # Initialize enhanced insulin pools (with backward compatibility)
    if "insulin_rapid_acting_units" in state.metabolic_state:
        insulin_pools = InsulinPools(
            rapid_acting=state.metabolic_state.get("insulin_rapid_acting_units", 0.1),
            basal=state.metabolic_state.get("insulin_basal_units", 0.4),
        )
    else:
        # Convert legacy single insulin value
        insulin_pools = InsulinPools.from_legacy(insulin_on_board)

    # Initialize insulin action buffer (with backward compatibility)
    if state.insulin_action_buffer_state is not None:
        buffer = InsulinActionBuffer(
            buffer=state.insulin_action_buffer_state,
            current_index=state.simulation_tick % 60,  # Use tick to track position
        )
    else:
        buffer = InsulinActionBuffer.create()

    activity_mode, activity_intensity = determine_activity(
        activity_timestamp,
        rng,
    )

    current_date = state.timestamp_utc.date()
    meal_flags_date = state.meal_flags_date
    daily_meal_flags = state.daily_meal_flags
    if meal_flags_date != current_date:
        daily_meal_flags = _fresh_daily_meal_flags()
        meal_flags_date = current_date

    meal_event, daily_meal_flags = maybe_trigger_meal_event(
        state.timestamp_utc,
        rng,
        daily_meal_flags,
        state.last_meal_time,
    )
    last_meal_time = state.last_meal_time
    bolus_insulin = 0.0
    if meal_event:
        meal_name, meal_carbs = meal_event
        carbs_in_stomach += meal_carbs
        last_meal_time = state.timestamp_utc
        
        # Calculate bolus insulin for meal
        bolus_insulin = calculate_bolus_insulin(
            carbs_grams=meal_carbs,
            current_glucose=current_glucose,
            rng=rng,
        )
        # Add bolus to rapid-acting pool
        insulin_pools = evolve_insulin_pools(
            pools=insulin_pools,
            add_basal=False,  # Don't add basal yet, we'll do it in enhanced model
            add_bolus=bolus_insulin,
        )

    # Use enhanced metabolic model
    glucose_true, insulin_pools, carbs_in_stomach = calculate_next_glucose_enhanced(
        current_glucose=current_glucose,
        insulin_pools=insulin_pools,
        insulin_action_buffer=buffer,
        carbs_in_stomach=carbs_in_stomach,
        sensitivity_factor=profile.insulin_sensitivity_factor,
        activity_intensity=activity_intensity,
        hour_of_day=hour_of_day,
        rng=rng,
    )
    
    # Update insulin_on_board for backward compatibility
    insulin_on_board = insulin_pools.to_legacy()

    vitals_target = calculate_vitals_target(
        profile=profile,
        glucose=glucose_true,
        activity_intensity=activity_intensity,
    )

    circadian_heart_rate = apply_circadian_drift(
        vitals_target["heart_rate_bpm"],
        hour_of_day,
        amplitude=5.0,
    )
    core_body_temp = apply_circadian_drift(36.6, hour_of_day, amplitude=0.5)

    ecg_wave, eda_wave = generate_waveform_snapshot(
        heart_rate_bpm=circadian_heart_rate,
    )

    updated_metabolic = dict(state.metabolic_state)
    updated_metabolic["glucose_true_mgdl"] = glucose_true
    updated_metabolic["insulin_on_board_units"] = insulin_on_board  # Legacy compatibility
    updated_metabolic["insulin_rapid_acting_units"] = insulin_pools.rapid_acting
    updated_metabolic["insulin_basal_units"] = insulin_pools.basal
    updated_metabolic["carbs_in_stomach_grams"] = carbs_in_stomach

    glucose_sensor = apply_sensor_noise(
        glucose_true + profile.cgm_noise_factor,
        activity_intensity,
        rng=rng,
    )
    glucose_sensor = (
        glucose_sensor if glucose_sensor is not None else float(glucose_true)
    )
    glucose_delta = glucose_sensor - current_glucose

    heart_rate_bpm = int(
        round(apply_sensor_noise(circadian_heart_rate, activity_intensity, rng=rng))
    )

    # SpO2: healthy individuals maintain 95-100% even during exercise
    # Slight decrease only at very high intensity, diabetics may have slightly lower baseline
    spo2_base = 97.0  # Slightly lower baseline for diabetic population
    spo2_exercise_drop = activity_intensity * 0.8  # Max ~0.8% drop at peak exercise
    spo2_pct = float(np.clip(spo2_base - spo2_exercise_drop, 94.0, 100.0))

    # Respiratory rate: 12-20 at rest, up to 30-40 during intense exercise
    resp_rate_rpm = int(round(14 + activity_intensity * 22))

    vitals_payload = {
        "heart_rate_bpm": heart_rate_bpm,
        "hrv_sdnn": float(
            apply_generic_sensor_noise(vitals_target["hrv_sdnn"], 1.0, rng=rng) or 0.0
        ),
        "qt_interval_ms": int(
            round(
                apply_generic_sensor_noise(
                    vitals_target["qt_interval_ms"],
                    sigma=5.0,
                    rng=rng,
                )
                or 0.0
            )
        ),
        "spo2_pct": spo2_pct,
        "resp_rate_rpm": resp_rate_rpm,
    }

    metabolics_payload = {
        "glucose_mgdl": int(round(glucose_sensor)),
        "trend_arrow": _trend_arrow(glucose_delta),
    }

    skin_temp_baseline = core_body_temp - 3.0 + activity_intensity * 0.6

    # Steps per minute: realistic cadence based on activity mode
    # Walking: ~80-100 steps/min, Running: ~140-180 steps/min
    # Use activity_intensity thresholds to determine gait
    if activity_intensity < 0.15:
        steps_per_min = 0  # Sedentary/sleeping
    elif activity_intensity < 0.4:
        # Light walking: 60-90 steps/min
        steps_per_min = int(round(60 + (activity_intensity - 0.15) * 120))
    elif activity_intensity < 0.6:
        # Brisk walking: 90-120 steps/min
        steps_per_min = int(round(90 + (activity_intensity - 0.4) * 150))
    else:
        # Running: 120-180 steps/min
        steps_per_min = int(round(120 + (activity_intensity - 0.6) * 150))

    wearable_payload = {
        "steps_per_minute": steps_per_min,
        "accel_y_g": float(
            apply_generic_sensor_noise(activity_intensity * 1.1, 0.05, rng=rng) or 0.0
        ),
        "skin_temp_c": float(
            apply_generic_sensor_noise(skin_temp_baseline, 0.2, rng=rng)
            or skin_temp_baseline
        ),
        "eda_microsiemens": float(
            apply_generic_sensor_noise(
                5.0 + eda_wave.mean() * 0.1 + vitals_target["eda_peak_flag"] * 3.0,
                0.3,
                rng=rng,
            )
            or 5.0
        ),
    }

    waveform_payload = {
        "ecg_short_window": ecg_wave.tolist(),
        "eda_short_window": eda_wave.tolist(),
    }

    meta_payload = {
        "message_id": str(uuid.uuid4()),
        "timestamp": state.timestamp_utc,
        "device_id": profile.patient_id,
    }

    new_state = replace(
        state,
        timestamp_utc=state.timestamp_utc + timedelta(minutes=1),
        simulation_tick=state.simulation_tick + 1,
        current_activity_mode=activity_mode,
        activity_intensity=activity_intensity,
        cumulative_fatigue=_update_fatigue(state.cumulative_fatigue, activity_intensity),
        metabolic_state=updated_metabolic,
        solver_internal_vector=[
            glucose_true,
            insulin_on_board,
            carbs_in_stomach,
        ],
        meal_flags_date=meal_flags_date,
        daily_meal_flags=daily_meal_flags,
        last_meal_time=last_meal_time,
        insulin_action_buffer_state=buffer.buffer.copy(),  # Save buffer state
    )

    payload = SensorPayload(
        meta=meta_payload,
        vitals=vitals_payload,
        metabolics=metabolics_payload,
        wearable=wearable_payload,
        waveform_snapshots=waveform_payload,
    )

    return new_state, payload


def evolve_patient_batch(
    profiles: Iterable[PatientStaticProfile],
    states: Iterable[PatientDynamicState],
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[PatientDynamicState, SensorPayload]]:
    """Advance multiple patients in lockstep minute increments."""

    rng = _default_rng(rng)
    outputs: List[Tuple[PatientDynamicState, SensorPayload]] = []
    for profile, state in zip(profiles, states):
        child_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        outputs.append(process_single_patient(profile, state, rng=child_rng))
    return outputs

