"""Cardiovascular/autonomic target calculations and waveform synthesis."""

from __future__ import annotations

from typing import Dict, Tuple

import neurokit2 as nk
import numpy as np

from data_models import PatientStaticProfile


def _calculate_qtc_bazett(heart_rate_bpm: float, base_qtc: float = 420.0) -> float:
    """Calculate QT interval using Bazett's formula.

    QTc = QT / sqrt(RR interval in seconds)
    Rearranged: QT = QTc * sqrt(RR)

    Normal QTc range: 350-450ms (males), 360-460ms (females)
    We use 420ms as baseline QTc for diabetic population (slightly elevated).
    """
    if heart_rate_bpm <= 0:
        return base_qtc

    rr_seconds = 60.0 / heart_rate_bpm
    qt_interval = base_qtc * np.sqrt(rr_seconds)

    # Clamp to physiological range (250-600ms)
    return float(np.clip(qt_interval, 250.0, 600.0))


def calculate_vitals_target(
    profile: PatientStaticProfile,
    glucose: float,
    activity_intensity: float,
) -> Dict[str, float]:
    """Couple metabolic stressors into cardiovascular targets.

    Models:
    - Heart rate: increases with activity (up to +110 bpm at max intensity)
    - HRV (SDNN): decreases with activity (sympathetic dominance)
    - QT interval: rate-corrected via Bazett's formula, prolonged during hypoglycemia
    - EDA: increases during hypoglycemic sympathetic surge
    """

    # Heart rate: resting baseline + activity component
    target_hr = profile.resting_hr_baseline + (activity_intensity * 110.0)

    # HRV decreases with activity due to vagal withdrawal
    # Minimum ~5ms SDNN during intense exercise
    hrv_sdnn = max(5.0, 55.0 - activity_intensity * 40.0)

    # Baseline QTc - slightly elevated for diabetic population
    base_qtc = 420.0
    eda_peak = 0.0

    # Hypoglycemia response: sympathetic surge
    if glucose < 70.0:
        hypoglycemia_severity = (70.0 - glucose) / 30.0  # Normalized 0-1 for 40-70 mg/dL
        hypoglycemia_severity = min(1.0, hypoglycemia_severity)

        # Tachycardia response to hypoglycemia
        target_hr += hypoglycemia_severity * 30.0

        # Reduced HRV during stress
        hrv_sdnn *= (1.0 - hypoglycemia_severity * 0.6)

        # QT prolongation during hypoglycemia (catecholamine effect)
        base_qtc += hypoglycemia_severity * 50.0

        # Sympathetic skin response (sweating)
        eda_peak = hypoglycemia_severity

    # Calculate actual QT interval from heart rate using Bazett's formula
    qt_interval = _calculate_qtc_bazett(target_hr, base_qtc)

    return {
        "heart_rate_bpm": float(target_hr),
        "hrv_sdnn": float(hrv_sdnn),
        "qt_interval_ms": float(qt_interval),
        "eda_peak_flag": float(eda_peak),
    }


def generate_waveform_snapshot(
    heart_rate_bpm: float,
    duration_seconds: int = 10,
    sampling_rate: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render ECG/EDA snippets suitable for downstream dashboards."""

    ecg = nk.ecg_simulate(
        duration=duration_seconds,
        heart_rate=max(30.0, heart_rate_bpm),
        sampling_rate=sampling_rate,
    )
    eda = nk.eda_simulate(
        duration=duration_seconds,
        sampling_rate=sampling_rate,
    )
    return np.asarray(ecg, dtype=float), np.asarray(eda, dtype=float)

