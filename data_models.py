"""Data layer definitions for the diabetic digital twin simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

import numpy as np


__all__ = [
    "ActivityMode",
    "Demographics",
    "PatientStaticProfile",
    "PatientDynamicState",
    "SensorPayload",
    "Sex",
]


class ActivityMode(str, Enum):
    """Discrete activity buckets aligned with the physiology rules engine."""

    SLEEP = "SLEEP"
    SEDENTARY = "SEDENTARY"
    WALKING = "WALKING"
    RUNNING = "RUNNING"
    STRESS_EVENT = "STRESS_EVENT"


class Sex(str, Enum):
    """Biological sex for physiological calculations."""

    MALE = "M"
    FEMALE = "F"


@dataclass(frozen=True)
class Demographics:
    """Derived demographic metrics computed from primary patient attributes.

    Encapsulates BMI classification, sex, and cardiovascular limits derived
    from anthropometric and age data using validated clinical formulas.
    """

    sex: Sex
    bmi: float
    max_heart_rate: float  # Tanaka formula: 208 - (0.7 * age)

    @property
    def bmi_category(self) -> str:
        """WHO BMI classification."""
        if self.bmi < 18.5:
            return "Underweight"
        elif self.bmi < 25.0:
            return "Normal"
        elif self.bmi < 30.0:
            return "Overweight"
        elif self.bmi < 35.0:
            return "Obese Class I"
        elif self.bmi < 40.0:
            return "Obese Class II"
        return "Obese Class III"

    @property
    def is_obese(self) -> bool:
        """Returns True if BMI >= 30 (clinical obesity threshold)."""
        return self.bmi >= 30.0


# =============================================================================
# Demographic Generation Functions (Single Responsibility Principle)
# =============================================================================


def _generate_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    min_val: float,
    max_val: float,
) -> float:
    """Sample from a truncated normal distribution via rejection sampling.

    Efficiently generates values within [min_val, max_val] by resampling
    until a valid value is obtained. For reasonable truncation bounds,
    this converges quickly.
    """
    while True:
        sample = rng.normal(mean, std)
        if min_val <= sample <= max_val:
            return float(sample)


def _generate_lognormal_bmi(
    rng: np.random.Generator,
    mode: float = 26.0,
    sigma: float = 0.30,
    min_bmi: float = 16.0,
    max_bmi: float = 50.0,
) -> float:
    """Generate BMI from a log-normal distribution with specified mode.

    Log-normal is appropriate for BMI because:
    1. BMI cannot be negative (log-normal has positive support)
    2. BMI distributions exhibit right-skew (obesity tail)
    3. Mode represents most common value in the population

    For log-normal: mode = exp(μ - σ²)
    Solving for μ: μ = ln(mode) + σ²

    Args:
        rng: NumPy random generator
        mode: Most frequent BMI value (default 26.0 for diabetic population)
        sigma: Shape parameter controlling spread (default 0.30)
        min_bmi: Lower physiological bound
        max_bmi: Upper physiological bound
    """
    mu = np.log(mode) + sigma ** 2
    while True:
        bmi = float(rng.lognormal(mu, sigma))
        if min_bmi <= bmi <= max_bmi:
            return bmi


def _calculate_max_heart_rate(age: int) -> float:
    """Calculate maximum heart rate using the Tanaka formula.

    Tanaka et al. (2001) showed this formula has better accuracy than
    the traditional 220-age formula across age groups:

        HRmax = 208 - (0.7 × age)

    Reference: Tanaka H, Monahan KD, Seals DR. Age-predicted maximal
    heart rate revisited. J Am Coll Cardiol. 2001;37(1):153-156.
    """
    return 208.0 - (0.7 * age)


def _calculate_resting_heart_rate(
    age: int,
    bmi: float,
    rng: np.random.Generator,
) -> float:
    """Calculate resting heart rate with physiological adjustments.

    Base resting HR of 60 bpm modified by:
    - Obesity penalty: +10 bpm if BMI >= 30 (increased cardiac workload)
    - Age adjustment: -5 bpm if age > 60 (parasympathetic tone changes)
    - Individual variation: ±8 bpm noise

    Returns value clamped to physiological range [40, 100] bpm.
    """
    base_hr = 60.0

    # Obesity increases resting HR due to elevated metabolic demand
    obesity_adjustment = 10.0 if bmi >= 30.0 else 0.0

    # Older adults often have lower resting HR due to autonomic changes
    age_adjustment = -5.0 if age > 60 else 0.0

    # Individual physiological variation
    noise = rng.normal(0.0, 8.0)

    resting_hr = base_hr + obesity_adjustment + age_adjustment + noise
    return float(np.clip(resting_hr, 40.0, 100.0))


def _calculate_insulin_sensitivity(age: int, bmi: float) -> float:
    """Calculate insulin sensitivity factor based on metabolic risk factors.

    Insulin sensitivity declines with:
    1. Adiposity (BMI > 25): -0.02 per BMI unit above 25
       - Adipose tissue inflammation impairs insulin signaling
    2. Age (> 30 years): -0.005 per year above 30
       - Sarcopenia and mitochondrial dysfunction reduce glucose uptake

    Normal sensitivity = 1.0
    Severe insulin resistance threshold = 0.1

    This simplified model captures the key epidemiological relationships
    between obesity, aging, and insulin resistance in T2DM.
    """
    sensitivity = 1.0

    # BMI penalty: adiposity-driven insulin resistance
    if bmi > 25.0:
        sensitivity -= 0.02 * (bmi - 25.0)

    # Age penalty: age-related metabolic decline
    if age > 30:
        sensitivity -= 0.005 * (age - 30)

    # Clamp to physiological bounds
    return float(np.clip(sensitivity, 0.1, 1.5))


# =============================================================================
# Patient Profile Dataclass
# =============================================================================


@dataclass(frozen=True)
class PatientStaticProfile:
    """Immutable traits that define a patient's physiology.

    Attributes are generated using cascading demographic logic where
    downstream attributes depend on upstream values (e.g., weight
    depends on height and BMI, resting HR depends on age and BMI).
    """

    patient_id: str
    age: int
    gender: str  # Kept as string for backward compatibility ("M" or "F")
    height_cm: float
    weight_kg: float
    resting_hr_baseline: float
    insulin_sensitivity_factor: float
    cgm_noise_factor: float
    demographics: Demographics

    @property
    def bmi(self) -> float:
        """Body Mass Index (delegated to Demographics for consistency)."""
        return self.demographics.bmi

    @property
    def max_heart_rate(self) -> float:
        """Maximum heart rate from Tanaka formula."""
        return self.demographics.max_heart_rate

    @staticmethod
    def generate_random(
        rng: Optional[np.random.Generator] = None,
    ) -> "PatientStaticProfile":
        """Generate a physiologically plausible patient using cascading demographics.

        Cascade Logic:
        1. Age & Sex: Independent base demographics
        2. Height: Sex-dependent normal distribution
        3. BMI: Log-normal (captures obesity tail), then derive weight
        4. Biological markers: Derived from age, sex, and BMI

        This approach ensures internally consistent patient profiles where
        correlated attributes (e.g., weight and insulin sensitivity via BMI)
        maintain epidemiologically valid relationships.
        """
        rng = rng or np.random.default_rng()
        patient_id = str(uuid.uuid4())

        # =====================================================================
        # CASCADE STEP 1: Age & Sex (Independent Base Demographics)
        # =====================================================================
        # Sex: 50/50 distribution
        sex = Sex.MALE if rng.random() < 0.5 else Sex.FEMALE
        gender = sex.value

        # Age: Truncated normal (18-85, μ=45, σ=15)
        # Diabetic population spans working adults to elderly
        age = int(round(_generate_truncated_normal(
            rng, mean=45.0, std=15.0, min_val=18.0, max_val=85.0
        )))

        # =====================================================================
        # CASCADE STEP 2: Height (Sex-Dependent)
        # =====================================================================
        # Population means from CDC/WHO anthropometric data
        # Males: μ=178cm (5'10"), Females: μ=163cm (5'4")
        height_mean = 178.0 if sex == Sex.MALE else 163.0
        height_std = 7.0  # Standard deviation for both sexes

        height_cm = _generate_truncated_normal(
            rng,
            mean=height_mean,
            std=height_std,
            min_val=140.0,  # ~4'7" - covers dwarfism
            max_val=210.0,  # ~6'11" - covers extreme tall
        )
        height_m = height_cm / 100.0

        # =====================================================================
        # CASCADE STEP 3: Weight via BMI (Log-Normal Distribution)
        # =====================================================================
        # Log-normal captures the right-skewed obesity distribution
        # Mode at 26.0 (slightly overweight - typical for pre-diabetic/diabetic)
        bmi = _generate_lognormal_bmi(rng, mode=26.0, sigma=0.30)

        # Derive weight from BMI and height: W = BMI × H²
        weight_kg = bmi * (height_m ** 2)
        weight_kg = float(np.clip(weight_kg, 35.0, 250.0))

        # Recalculate BMI from clamped weight for consistency
        bmi = weight_kg / (height_m ** 2)

        # =====================================================================
        # CASCADE STEP 4: Biological Derivations
        # =====================================================================
        # Maximum heart rate: Tanaka formula (validated across age groups)
        max_hr = _calculate_max_heart_rate(age)

        # Resting heart rate: Base + obesity/age adjustments
        resting_hr_baseline = _calculate_resting_heart_rate(age, bmi, rng)

        # Insulin sensitivity: Derived from BMI and age risk factors
        insulin_sensitivity_factor = _calculate_insulin_sensitivity(age, bmi)

        # CGM noise factor: Sensor quality variation (1.0 = baseline)
        # Log-normal ensures positive values with slight right skew
        cgm_noise_factor = float(np.clip(
            rng.lognormal(mean=0.0, sigma=0.15),
            0.5,
            2.0
        ))

        # =====================================================================
        # Assemble Demographics Object
        # =====================================================================
        demographics = Demographics(
            sex=sex,
            bmi=float(bmi),
            max_heart_rate=float(max_hr),
        )

        return PatientStaticProfile(
            patient_id=patient_id,
            age=age,
            gender=gender,
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            resting_hr_baseline=float(resting_hr_baseline),
            insulin_sensitivity_factor=float(insulin_sensitivity_factor),
            cgm_noise_factor=float(cgm_noise_factor),
            demographics=demographics,
        )


def _default_meal_flags() -> Dict[str, bool]:
    return {"breakfast": False, "lunch": False, "dinner": False}


def _default_metabolic_state() -> Dict[str, Union[float, List[float]]]:
    """Initialize metabolic state with realistic fasting values.

    - Fasting glucose: ~100-110 mg/dL (slightly elevated for diabetics)
    - Insulin on board: ~0.5 units (residual from basal infusion)
    - Carbs in stomach: 0.0 (fasting state)
    """
    return {
        "glucose_true_mgdl": 105.0,
        "insulin_on_board_units": 0.5,
        "carbs_in_stomach_grams": 0.0,
        # Enhanced model fields (optional, for backward compatibility)
        "insulin_rapid_acting_units": 0.1,
        "insulin_basal_units": 0.4,
    }


@dataclass
class PatientDynamicState:
    """Minute-level snapshot of a patient flowing through the master driver."""

    timestamp_utc: datetime
    simulation_tick: int
    current_activity_mode: ActivityMode
    activity_intensity: float  # Normalized 0.0-1.0
    cumulative_fatigue: float  # Normalized 0.0-1.0
    metabolic_state: Dict[str, Union[float, List[float]]] = field(default_factory=_default_metabolic_state)
    solver_internal_vector: List[float] = field(
        default_factory=lambda: [105.0, 0.5, 0.0]  # Match initial metabolic state
    )
    meal_flags_date: date = field(default_factory=date.today)
    daily_meal_flags: Dict[str, bool] = field(default_factory=_default_meal_flags)
    last_meal_time: Optional[datetime] = None


@dataclass
class SensorPayload:
    """BigQuery-aligned payload emitted to Pub/Sub."""

    meta: Dict[str, Union[str, datetime]]
    vitals: Dict[str, Union[int, float]]
    metabolics: Dict[str, Union[int, float, str]]
    wearable: Dict[str, Union[int, float]]
    waveform_snapshots: Dict[str, Union[List[float], Dict[str, Any]]]

    def to_dict(self, include_waveforms: bool = True) -> Dict[str, Any]:
        """Serialize payload; optional waveform suppression for Pub/Sub."""

        def _serialize(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {k: _serialize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_serialize(v) for v in value]
            return value

        serialized = {
            "meta": _serialize(self.meta),
            "vitals": _serialize(self.vitals),
            "metabolics": _serialize(self.metabolics),
            "wearable": _serialize(self.wearable),
        }
        if include_waveforms:
            serialized["waveform_snapshots"] = _serialize(self.waveform_snapshots)
        return serialized


