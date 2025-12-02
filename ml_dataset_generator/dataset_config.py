"""Configuration for ML dataset generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DatasetConfig:
    """Configuration parameters for ML dataset generation.
    
    Follows Single Responsibility Principle: Only manages configuration.
    """
    
    # Patient generation
    num_patients: int = 1000
    simulation_days_per_patient: int = 3
    
    # Time series configuration  
    window_size_minutes: int = 5  # Number of timesteps (t-4, t-3, t-2, t-1, t0)
    
    # Output configuration
    output_dir: Path = Path("ml_dataset_output")
    train_split: float = 0.70
    val_split: float = 0.15
    # test_split is implicit: 1.0 - train_split - val_split = 0.15
    
    # Performance tuning
    batch_size: int = 50  # Process 50 patients at a time to manage memory
    random_seed: int = 42
    
    # Features to extract from simulation (per timestep)
    features_to_extract: List[str] = field(default_factory=lambda: [
        "glucose_mgdl",
        "heart_rate_bpm",
        "hrv_sdnn",
        "respiratory_rate_rpm",
        "spo2_pct",
        "steps_per_minute",
        "vertical_acceleration_g",
        "skin_temperature_c",
        "eda_microsiemens",
        "qt_interval_ms",
        "insulin_on_board",
        "carbs_in_stomach",
        "activity_intensity",
        "hour_of_day",
    ])
    
    # Regression targets to predict (at t+1)
    regression_targets: List[str] = field(default_factory=lambda: [
        "glucose_mgdl",
        "heart_rate_bpm",
        "hrv_sdnn",
        "respiratory_rate_rpm",
    ])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output directory is a Path object
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        
        # Validate splits
        total_split = self.train_split + self.val_split
        if not (0.0 < total_split < 1.0):
            raise ValueError(
                f"Train + val splits must be between 0 and 1, got {total_split}"
            )
    
    @property
    def test_split(self) -> float:
        """Calculate test split from train and val splits."""
        return 1.0 - self.train_split - self.val_split
    
    @property
    def expected_samples_per_patient(self) -> int:
        """Calculate expected samples per patient.
        
        Each day = 1440 minutes
        With window size of 5, we need 5 minutes of history + 1 for target
        So we get approximately (1440 - window_size - 1) samples per day
        """
        minutes_per_day = 1440
        usable_minutes_per_day = minutes_per_day - self.window_size_minutes
        return usable_minutes_per_day * self.simulation_days_per_patient
    
    @property
    def expected_total_samples(self) -> int:
        """Calculate expected total samples."""
        return self.num_patients * self.expected_samples_per_patient
    
    def summary(self) -> str:
        """Generate a summary of the configuration."""
        return f"""
╔═══════════════════════════════════════════════════════════╗
║         ML Dataset Generation Configuration               ║
╚═══════════════════════════════════════════════════════════╝

Patients:
  - Total: {self.num_patients:,}
  - Simulation Days: {self.simulation_days_per_patient}
  - Batch Size: {self.batch_size} patients

Time Series:
  - Window Size: {self.window_size_minutes} timesteps
  - Features per Timestep: {len(self.features_to_extract)}
  - Total Input Features: {len(self.features_to_extract) * self.window_size_minutes}

Expected Output:
  - Samples per Patient: ~{self.expected_samples_per_patient:,}
  - Total Samples: ~{self.expected_total_samples:,}

Data Splits:
  - Training: {self.train_split:.1%} (~{int(self.expected_total_samples * self.train_split):,} samples)
  - Validation: {self.val_split:.1%} (~{int(self.expected_total_samples * self.val_split):,} samples)
  - Testing: {self.test_split:.1%} (~{int(self.expected_total_samples * self.test_split):,} samples)

Regression Targets: {len(self.regression_targets)}
  {', '.join(self.regression_targets)}

Classification Labels: 6
  - Hypoglycemia Risk
  - Hyperglycemia Risk
  - Fall Risk
  - Cardiac Anomaly
  - Severe Hypotension Risk
  - Autonomic Dysregulation

Output Directory: {self.output_dir.absolute()}
Random Seed: {self.random_seed}
"""

