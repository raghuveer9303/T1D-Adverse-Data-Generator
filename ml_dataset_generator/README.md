# ML Dataset Generator for Diabetic Digital Twin

This package generates comprehensive time-series datasets for machine learning, specifically designed for Apache Spark MLlib training.

## Overview

The generator creates realistic medical sensor data from simulated diabetic patients with:
- **1000 unique patients** with diverse demographics
- **~3 million time-series samples** (3 days per patient)
- **5-minute sliding windows** for temporal pattern learning
- **14 physiological features** per timestep
- **4 regression targets** (glucose, heart rate, HRV, respiratory rate)
- **6 classification labels** for adverse event detection

## Architecture (SOLID & OOP)

The generator follows SOLID principles with clear separation of concerns:

```
generate_ml_dataset.py          # Main entry point
│
ml_dataset_generator/
├── dataset_config.py            # Configuration management (SRP)
├── patient_generator.py         # Patient profile generation (SRP)
├── simulation_runner.py         # Simulation execution (SRP)
├── window_extractor.py          # Time-series window extraction (SRP)
├── label_calculator.py          # Adverse event labeling (SRP)
├── dataset_writer.py            # Parquet file I/O (SRP)
└── orchestrator.py              # Pipeline coordination (DIP, OCP)
```

### Design Principles

- **Single Responsibility Principle (SRP)**: Each class has one clear purpose
- **Open/Closed Principle (OCP)**: Extensible without modification
- **Dependency Inversion Principle (DIP)**: Depends on abstractions, not concretions
- **No interfaces** (as requested): Uses concrete classes with clear contracts

## Output Structure

The generator produces two tables optimized for Spark joins:

### 1. Patient Demographics Table
**File**: `patient_demographics.parquet`

| Column | Type | Description |
|--------|------|-------------|
| patient_id | string | Primary key |
| age | int | Patient age (years) |
| gender | string | M/F |
| bmi | double | Body Mass Index |
| height_cm | double | Height in cm |
| weight_kg | double | Weight in kg |
| resting_hr_baseline | double | Baseline resting HR |
| max_heart_rate | double | Max HR (Tanaka formula) |
| insulin_sensitivity_factor | double | Insulin sensitivity (0.1-1.5) |

**Rows**: 1000 (one per patient)

### 2. Time-Series Features Tables
**Files**: 
- `timeseries_features_train.parquet` (70%)
- `timeseries_features_val.parquet` (15%)
- `timeseries_features_test.parquet` (15%)

**Schema**:
- `patient_id`: Foreign key to demographics
- `timestamp`: Window end time
- `feature_t0_glucose`: Current glucose (t=0)
- `feature_t0_heart_rate`: Current heart rate
- ... (14 features × 5 timesteps = 70 feature columns)
- `feature_t4_glucose`: Glucose 4 minutes ago
- `feature_t4_heart_rate`: Heart rate 4 minutes ago
- `target_glucose_mgdl`: Next glucose (t+1)
- `target_heart_rate_bpm`: Next heart rate (t+1)
- `target_hrv_sdnn`: Next HRV (t+1)
- `target_respiratory_rate_rpm`: Next respiratory rate (t+1)
- `label_hypoglycemia_risk`: Binary (0/1)
- `label_hyperglycemia_risk`: Binary (0/1)
- `label_fall_risk`: Binary (0/1)
- `label_cardiac_anomaly`: Binary (0/1)
- `label_severe_hypotension_risk`: Binary (0/1)
- `label_autonomic_dysregulation`: Binary (0/1)

**Total Rows**: ~3,000,000

## Features Extracted (Per Timestep)

1. **glucose_mgdl** - Blood glucose level
2. **heart_rate_bpm** - Heart rate
3. **hrv_sdnn** - Heart rate variability (SDNN)
4. **respiratory_rate_rpm** - Breathing rate
5. **spo2_pct** - Oxygen saturation
6. **steps_per_minute** - Activity level
7. **vertical_acceleration_g** - Movement intensity
8. **skin_temperature_c** - Skin temperature
9. **eda_microsiemens** - Electrodermal activity (stress)
10. **qt_interval_ms** - Cardiac repolarization
11. **insulin_on_board** - Active insulin
12. **carbs_in_stomach** - Digesting carbohydrates
13. **activity_intensity** - Normalized activity (0-1)
14. **hour_of_day** - Circadian context

## Adverse Event Labels

Each label is calculated using physiologically-grounded rules:

### 1. Hypoglycemia Risk
- **Rule**: glucose < 80 mg/dL AND trending down (Δ < -3 mg/dL/min)
- **OR**: glucose < 70 mg/dL (severe)
- **Clinical significance**: Risk of loss of consciousness, seizures

### 2. Hyperglycemia Risk
- **Rule**: glucose > 180 mg/dL AND trending up (Δ > 5 mg/dL/min)
- **OR**: glucose > 250 mg/dL (severe)
- **Clinical significance**: Diabetic ketoacidosis risk

### 3. Fall Risk
- **Rule**: (glucose < 70 mg/dL) AND (high activity > 0.5) AND (low HRV < 20 ms)
- **Clinical significance**: Dizziness/coordination loss during activity

### 4. Cardiac Anomaly
- **Rule**: QTc > 470ms OR HR < 45 OR HR > 160 (at rest)
- **Clinical significance**: Arrhythmia risk, cardiac stress

### 5. Severe Hypotension Risk
- **Rule**: glucose < 70 AND activity > 0.3 AND HR_delta < -10 bpm
- **Clinical significance**: Blood pressure crash risk

### 6. Autonomic Dysregulation
- **Rule**: EDA > 12 μS AND HRV < 15 ms AND glucose extremes
- **Clinical significance**: Autonomic nervous system stress

## Usage

### Basic Usage

```bash
python generate_ml_dataset.py
```

This will generate the full dataset with default configuration:
- 1000 patients
- 3 days per patient
- 5-minute windows
- Output to `./ml_dataset_output/`
- **Parallel processing**: Uses all CPU cores automatically
- **Append mode**: Enabled by default - can run multiple times to grow dataset
- **Logging**: Console output + timestamped log files in output directory

### Features

#### 🚀 Parallel Processing
The generator automatically uses **all available CPU cores** for maximum performance:
- Patient simulations run in parallel
- Window extraction parallelized across patients
- Label calculation parallelized in chunks
- Non-blocking execution with real-time progress tracking

#### 📝 Comprehensive Logging
- **Console output**: Real-time progress with colored indicators
- **Log files**: Detailed timestamped logs saved to `ml_dataset_output/ml_dataset_generation_YYYYMMDD_HHMMSS.log`
- **Progress tracking**: See which step is running and completion percentage
- **Performance metrics**: Track samples generated, file sizes, and timing

#### 💾 Append Mode
The generator supports **incremental dataset generation**:
- Run multiple times to add more patients/data
- Existing files are appended to (not overwritten)
- Useful for:
  - Growing datasets over time
  - Recovering from interruptions
  - Distributed generation across machines

### Using in Spark MLlib

```python
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("Diabetic ML Model") \
    .getOrCreate()

# Load datasets
demographics = spark.read.parquet("ml_dataset_output/patient_demographics.parquet")
train = spark.read.parquet("ml_dataset_output/timeseries_features_train.parquet")
val = spark.read.parquet("ml_dataset_output/timeseries_features_val.parquet")

# Join demographics with time-series
train_full = train.join(demographics, on="patient_id", how="left")

# Prepare features for MLlib
from pyspark.ml.feature import VectorAssembler

feature_cols = [col for col in train_full.columns if col.startswith("feature_t")]
assembler = VectorAssembler(inputCols=feature_cols + ["age", "bmi", "insulin_sensitivity_factor"], 
                            outputCol="features")
train_ml = assembler.transform(train_full)

# Train regression model (example: glucose prediction)
from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(featuresCol="features", labelCol="target_glucose_mgdl")
model = rf.fit(train_ml)

# Train classification model (example: hypoglycemia detection)
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(featuresCol="features", labelCol="label_hypoglycemia_risk")
clf_model = gbt.fit(train_ml)
```

## Performance

- **Generation time**: ~10-20 minutes for full dataset (1000 patients) with parallel processing
- **Memory usage**: ~4-8 GB peak (batched processing)
- **Output size**: ~2-3 GB (compressed Parquet)
- **Parallelization**: **FULL MULTICORE SUPPORT** - Uses all CPU cores by default
  - Parallel patient simulation execution
  - Parallel window extraction
  - Parallel label calculation
  - Non-blocking execution with progress tracking
- **Logging**: Comprehensive console and file logging for progress tracking

## Configuration

Edit `generate_ml_dataset.py` to customize:

```python
config = DatasetConfig(
    num_patients=1000,              # Number of unique patients
    simulation_days_per_patient=3,  # Days to simulate per patient
    window_size_minutes=5,           # Timesteps per window
    output_dir=Path("output"),       # Output directory
    train_split=0.70,                # Training split
    val_split=0.15,                  # Validation split
    batch_size=50,                   # Patients per batch
    random_seed=42,                  # For reproducibility
)

# Run with custom parallel processing settings
orchestrator.generate_full_dataset(
    use_parallel=True,    # Enable/disable parallel processing
    n_jobs=8,             # Number of CPU cores (None = use all)
    append_mode=True,     # Enable/disable file appending
)
```

### Advanced Configuration

**Disable Parallel Processing** (for debugging):
```python
orchestrator.generate_full_dataset(
    use_parallel=False,   # Sequential mode
    append_mode=True,
)
```

**Control CPU Usage**:
```python
orchestrator.generate_full_dataset(
    use_parallel=True,
    n_jobs=4,             # Use only 4 cores
    append_mode=True,
)
```

**Overwrite Mode** (clear existing data):
```python
orchestrator.generate_full_dataset(
    use_parallel=True,
    n_jobs=None,          # Use all cores
    append_mode=False,    # Overwrite existing files
)
```

## Requirements

- Python 3.8+
- numpy
- pandas
- pyarrow
- Existing simulation dependencies (see main requirements.txt)

## License

Part of the Diabetic Digital Twin project.

