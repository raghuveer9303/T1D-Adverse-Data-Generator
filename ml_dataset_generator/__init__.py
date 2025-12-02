"""ML Dataset Generator for Diabetic Digital Twin.

This package generates time-series datasets suitable for Spark MLlib training,
with separate tables for patient demographics and time-series features.
"""

__version__ = "1.0.0"

__all__ = [
    "DatasetConfig",
    "PatientGenerator",
    "SimulationRunner",
    "WindowExtractor",
    "LabelCalculator",
    "DatasetWriter",
    "MLDatasetOrchestrator",
]

