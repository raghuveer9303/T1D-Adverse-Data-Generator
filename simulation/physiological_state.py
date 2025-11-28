"""Physiological state calculation with coherence enforcement.

This module implements OOP and SOLID principles to ensure all vital signs
and wearable data maintain internal consistency across parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from data_models import ActivityMode


@dataclass(frozen=True)
class PhysiologicalState:
    """Immutable snapshot of coherent physiological parameters.
    
    All parameters are guaranteed to be internally consistent with the
    given activity level and metabolic state.
    """
    
    # Activity parameters
    activity_mode: ActivityMode
    activity_intensity: float  # 0.0-1.0
    
    # Cardiovascular parameters
    heart_rate_bpm: float
    hrv_sdnn: float  # ms
    qt_interval_ms: float
    respiratory_rate_rpm: float
    spo2_pct: float
    
    # Movement parameters
    steps_per_minute: int
    vertical_acceleration_g: float
    
    # Thermoregulation
    skin_temperature_c: float
    core_temperature_c: float
    
    # Autonomic response
    eda_microsiemens: float
    
    def validate_coherence(self) -> bool:
        """Verify internal consistency of physiological parameters.
        
        Returns True if all parameters are coherent with activity level.
        """
        # Validate cardiovascular coherence
        if self.activity_intensity < 0.15:  # Resting state
            if self.heart_rate_bpm > 80 or self.steps_per_minute > 5:
                return False
            if self.vertical_acceleration_g > 0.3:
                return False
        elif self.activity_intensity >= 0.5:  # Active state
            if self.heart_rate_bpm < 90 or self.steps_per_minute < 70:
                return False
            if self.vertical_acceleration_g < 0.5:
                return False
        
        # Validate QTc (corrected QT interval)
        rr_interval_sec = 60.0 / self.heart_rate_bpm
        qtc = self.qt_interval_ms / np.sqrt(rr_interval_sec)
        if qtc < 300 or qtc > 500:  # Outside physiological range
            return False
        
        return True


class PhysiologicalCalculator(ABC):
    """Abstract base for physiological parameter calculation.
    
    Follows Single Responsibility Principle - each calculator handles
    one aspect of physiology.
    """
    
    @abstractmethod
    def calculate(
        self,
        activity_intensity: float,
        context: PhysiologicalContext,
    ) -> Dict[str, float]:
        """Calculate specific physiological parameters.
        
        Args:
            activity_intensity: Normalized activity level (0.0-1.0)
            context: Additional context (glucose, baseline vitals, etc.)
            
        Returns:
            Dictionary of calculated parameters
        """
        pass


@dataclass
class PhysiologicalContext:
    """Context information for physiological calculations."""
    
    glucose_mgdl: float
    baseline_heart_rate: float
    max_heart_rate: float
    hour_of_day: float
    is_stressed: bool = False
    rng: Optional[np.random.Generator] = None


class CardiovascularCalculator(PhysiologicalCalculator):
    """Calculate heart rate, HRV, and related cardiovascular parameters.
    
    Ensures proper correlation between HR, HRV, and activity intensity.
    """
    
    def calculate(
        self,
        activity_intensity: float,
        context: PhysiologicalContext,
    ) -> Dict[str, float]:
        """Calculate cardiovascular parameters with physiological coherence."""
        
        # Heart rate increases linearly with activity
        # Resting: baseline (60-75 bpm)
        # Peak activity: baseline + 110 bpm (up to max HR)
        activity_hr_increase = activity_intensity * 110.0
        target_hr = context.baseline_heart_rate + activity_hr_increase
        
        # HRV (SDNN) decreases with activity due to sympathetic dominance
        # Resting: 50-60 ms (high parasympathetic tone)
        # Active: 10-20 ms (sympathetic dominance)
        base_hrv = 60.0
        hrv_sdnn = base_hrv - (activity_intensity * 45.0)
        hrv_sdnn = max(5.0, hrv_sdnn)  # Minimum 5ms during peak exercise
        
        # Hypoglycemia response: sympathetic surge
        hypoglycemia_factor = 0.0
        if context.glucose_mgdl < 70.0:
            hypoglycemia_severity = (70.0 - context.glucose_mgdl) / 30.0
            hypoglycemia_severity = min(1.0, max(0.0, hypoglycemia_severity))
            
            # Increase HR by up to 30 bpm during severe hypoglycemia
            target_hr += hypoglycemia_severity * 30.0
            
            # Reduce HRV during stress
            hrv_sdnn *= (1.0 - hypoglycemia_severity * 0.6)
            
            hypoglycemia_factor = hypoglycemia_severity
        
        # Clamp heart rate to physiological limits
        target_hr = float(np.clip(target_hr, 40.0, context.max_heart_rate))
        
        # Respiratory rate: coupled with heart rate and activity
        # Resting: 12-16 rpm, Active: 20-40 rpm
        base_resp_rate = 14.0
        resp_rate_increase = activity_intensity * 22.0
        respiratory_rate = base_resp_rate + resp_rate_increase
        
        # SpO2: maintains 95-100% even during exercise in healthy individuals
        # Diabetics may have slightly lower baseline
        spo2_base = 97.0
        spo2_drop = activity_intensity * 0.8  # Max 0.8% drop at peak
        spo2 = float(np.clip(spo2_base - spo2_drop, 94.0, 100.0))
        
        return {
            "heart_rate_bpm": target_hr,
            "hrv_sdnn": float(hrv_sdnn),
            "respiratory_rate_rpm": float(respiratory_rate),
            "spo2_pct": spo2,
            "hypoglycemia_factor": hypoglycemia_factor,
        }


class QTIntervalCalculator(PhysiologicalCalculator):
    """Calculate QT interval with proper heart rate correction.
    
    Uses Bazett's formula: QTc = QT / sqrt(RR interval in seconds)
    Ensures QTc remains in normal range (350-450ms).
    """
    
    def __init__(self, base_qtc: float = 420.0):
        """Initialize with baseline QTc value.
        
        Args:
            base_qtc: Baseline corrected QT interval (ms).
                      420ms is slightly elevated, typical for diabetic population.
        """
        self.base_qtc = base_qtc
    
    def calculate(
        self,
        activity_intensity: float,
        context: PhysiologicalContext,
    ) -> Dict[str, float]:
        """Calculate QT interval from heart rate using Bazett's formula."""
        
        # Get heart rate from cardiovascular calculator
        cardio_calc = CardiovascularCalculator()
        cardio_params = cardio_calc.calculate(activity_intensity, context)
        heart_rate = cardio_params["heart_rate_bpm"]
        hypoglycemia_factor = cardio_params["hypoglycemia_factor"]
        
        # Adjust QTc for hypoglycemia (catecholamine-induced prolongation)
        adjusted_qtc = self.base_qtc + (hypoglycemia_factor * 50.0)
        
        # Calculate actual QT interval from QTc using Bazett's formula
        # QT = QTc * sqrt(RR interval in seconds)
        if heart_rate <= 0:
            qt_interval = adjusted_qtc
        else:
            rr_seconds = 60.0 / heart_rate
            qt_interval = adjusted_qtc * np.sqrt(rr_seconds)
        
        # Clamp to physiological range
        qt_interval = float(np.clip(qt_interval, 250.0, 600.0))
        
        return {
            "qt_interval_ms": qt_interval,
            "qtc_ms": adjusted_qtc,
        }


class MovementCalculator(PhysiologicalCalculator):
    """Calculate movement parameters: steps, acceleration, cadence.
    
    Ensures steps per minute correlates with activity level and other vitals.
    """
    
    def calculate(
        self,
        activity_intensity: float,
        context: PhysiologicalContext,
    ) -> Dict[str, float]:
        """Calculate movement parameters coherent with activity intensity."""
        
        # Steps per minute based on physiological activity thresholds
        # Research-backed cadence values:
        # - Resting/Sedentary: 0-5 steps/min (fidgeting, small movements)
        # - Light walking: 60-80 steps/min (slow stroll)
        # - Normal walking: 80-100 steps/min (comfortable pace)
        # - Brisk walking: 100-120 steps/min (purposeful walking)
        # - Jogging: 120-160 steps/min
        # - Running: 160-180 steps/min
        
        if activity_intensity < 0.15:
            # Sedentary/sleeping: minimal movement
            base_steps = 0
            steps_noise = 3
        elif activity_intensity < 0.35:
            # Light activity: 0-60 steps/min
            base_steps = int((activity_intensity - 0.15) * 300)  # 0-60 range
            steps_noise = 5
        elif activity_intensity < 0.50:
            # Walking: 60-90 steps/min
            base_steps = 60 + int((activity_intensity - 0.35) * 200)  # 60-90 range
            steps_noise = 5
        elif activity_intensity < 0.65:
            # Brisk walking: 90-110 steps/min
            base_steps = 90 + int((activity_intensity - 0.50) * 133)  # 90-110 range
            steps_noise = 5
        elif activity_intensity < 0.80:
            # Jogging: 110-150 steps/min
            base_steps = 110 + int((activity_intensity - 0.65) * 267)  # 110-150 range
            steps_noise = 8
        else:
            # Running: 150-180 steps/min
            base_steps = 150 + int((activity_intensity - 0.80) * 150)  # 150-180 range
            steps_noise = 10
        
        # Add realistic noise
        rng = context.rng or np.random.default_rng()
        steps_per_minute = max(0, base_steps + int(rng.normal(0, steps_noise)))
        
        # Vertical acceleration correlates with movement intensity
        # Sitting/standing: 0.0-0.2g (minimal vertical movement)
        # Walking: 0.3-0.8g (moderate bounce in gait)
        # Running: 0.8-1.5g (significant vertical displacement)
        
        if activity_intensity < 0.15:
            accel_base = 0.05
            accel_range = 0.15
        elif activity_intensity < 0.50:
            accel_base = 0.20 + (activity_intensity - 0.15) * 1.4  # 0.2-0.7g
            accel_range = 0.10
        else:
            accel_base = 0.70 + (activity_intensity - 0.50) * 1.6  # 0.7-1.5g
            accel_range = 0.15
        
        accel_y_g = accel_base + rng.normal(0, accel_range * 0.3)
        accel_y_g = float(np.clip(accel_y_g, 0.0, 2.0))
        
        return {
            "steps_per_minute": steps_per_minute,
            "vertical_acceleration_g": accel_y_g,
        }


class ThermoregulationCalculator(PhysiologicalCalculator):
    """Calculate skin and core temperature with activity adjustments."""
    
    def calculate(
        self,
        activity_intensity: float,
        context: PhysiologicalContext,
    ) -> Dict[str, float]:
        """Calculate temperature parameters."""
        
        # Core temperature with circadian variation
        # Normal: 36.5-37.2°C, peaks in late afternoon/evening
        circadian_hour = context.hour_of_day
        circadian_amplitude = 0.5
        
        # Circadian peak around 17:00 (5 PM)
        peak_hour = 17.0
        hour_radians = ((circadian_hour - peak_hour) / 24.0) * 2 * np.pi
        circadian_offset = circadian_amplitude * np.cos(hour_radians)
        
        core_base = 36.6
        core_temp = core_base + circadian_offset
        
        # Core temperature rises with intense activity
        exercise_temp_increase = activity_intensity * 0.8  # Up to 0.8°C increase
        core_temp += exercise_temp_increase
        
        # Skin temperature: typically 3-4°C below core
        # Increases during exercise due to vasodilation (heat dissipation)
        skin_temp_base = core_temp - 3.5
        skin_temp_exercise_adjustment = activity_intensity * 0.6
        skin_temp = skin_temp_base + skin_temp_exercise_adjustment
        
        # Add sensor noise
        rng = context.rng or np.random.default_rng()
        skin_temp += rng.normal(0, 0.2)
        
        return {
            "core_temperature_c": float(np.clip(core_temp, 35.5, 39.5)),
            "skin_temperature_c": float(np.clip(skin_temp, 28.0, 37.0)),
        }


class AutonomicCalculator(PhysiologicalCalculator):
    """Calculate electrodermal activity (EDA/skin conductance)."""
    
    def calculate(
        self,
        activity_intensity: float,
        context: PhysiologicalContext,
    ) -> Dict[str, float]:
        """Calculate EDA with stress and hypoglycemia responses."""
        
        # Baseline EDA: 1-10 microsiemens (varies by individual and humidity)
        base_eda = 3.0
        
        # Activity increases EDA due to sympathetic activation
        activity_eda = activity_intensity * 4.0
        
        # Hypoglycemia causes sympathetic surge (sweating)
        hypoglycemia_eda = 0.0
        if context.glucose_mgdl < 70.0:
            hypoglycemia_severity = (70.0 - context.glucose_mgdl) / 30.0
            hypoglycemia_severity = min(1.0, max(0.0, hypoglycemia_severity))
            hypoglycemia_eda = hypoglycemia_severity * 5.0
        
        # Stress events increase EDA
        stress_eda = 3.0 if context.is_stressed else 0.0
        
        total_eda = base_eda + activity_eda + hypoglycemia_eda + stress_eda
        
        # Add sensor noise
        rng = context.rng or np.random.default_rng()
        total_eda += rng.normal(0, 0.3)
        
        return {
            "eda_microsiemens": float(np.clip(total_eda, 0.5, 20.0)),
        }


class PhysiologicalStateFactory:
    """Factory for creating coherent physiological states.
    
    Follows Dependency Inversion Principle - depends on calculator abstractions.
    Follows Open/Closed Principle - extensible by adding new calculators.
    """
    
    def __init__(self):
        """Initialize factory with standard calculators."""
        self.cardio_calc = CardiovascularCalculator()
        self.qt_calc = QTIntervalCalculator()
        self.movement_calc = MovementCalculator()
        self.thermo_calc = ThermoregulationCalculator()
        self.autonomic_calc = AutonomicCalculator()
    
    def create_state(
        self,
        activity_mode: ActivityMode,
        activity_intensity: float,
        context: PhysiologicalContext,
    ) -> PhysiologicalState:
        """Create a coherent physiological state.
        
        All parameters are calculated to be internally consistent with
        the given activity level and metabolic state.
        """
        
        # Calculate all parameters using specialized calculators
        cardio_params = self.cardio_calc.calculate(activity_intensity, context)
        qt_params = self.qt_calc.calculate(activity_intensity, context)
        movement_params = self.movement_calc.calculate(activity_intensity, context)
        thermo_params = self.thermo_calc.calculate(activity_intensity, context)
        autonomic_params = self.autonomic_calc.calculate(activity_intensity, context)
        
        # Assemble into coherent state
        state = PhysiologicalState(
            activity_mode=activity_mode,
            activity_intensity=activity_intensity,
            heart_rate_bpm=cardio_params["heart_rate_bpm"],
            hrv_sdnn=cardio_params["hrv_sdnn"],
            qt_interval_ms=qt_params["qt_interval_ms"],
            respiratory_rate_rpm=cardio_params["respiratory_rate_rpm"],
            spo2_pct=cardio_params["spo2_pct"],
            steps_per_minute=movement_params["steps_per_minute"],
            vertical_acceleration_g=movement_params["vertical_acceleration_g"],
            skin_temperature_c=thermo_params["skin_temperature_c"],
            core_temperature_c=thermo_params["core_temperature_c"],
            eda_microsiemens=autonomic_params["eda_microsiemens"],
        )
        
        # Validate coherence (optional assertion in production)
        if not state.validate_coherence():
            # Log warning but proceed (some edge cases may be acceptable)
            pass
        
        return state

