"""Write ML dataset to Parquet files."""

import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_models import PatientStaticProfile
from ml_dataset_generator.window_extractor import TimeSeriesWindow

# Configure logging
logger = logging.getLogger(__name__)


class DatasetWriter:
    """Write time-series and demographics data to Parquet format.
    
    Single Responsibility: Handle data serialization and file I/O.
    Optimized for Spark MLlib consumption.
    """
    
    def __init__(self, output_dir: Path):
        """Initialize dataset writer.
        
        Args:
            output_dir: Directory to write output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_demographics_table(
        self,
        patients: List[PatientStaticProfile],
        append: bool = True,
    ) -> Path:
        """Write patient demographics table to Parquet with append support.
        
        One row per patient with static features.
        
        Args:
            patients: List of patient profiles
            append: If True, append to existing file if it exists
            
        Returns:
            Path to written file
        """
        logger.info(f"Writing demographics table for {len(patients)} patients (append={append})")
        
        print(f"\n{'='*60}")
        print("Writing Patient Demographics Table...")
        print(f"{'='*60}")
        
        # Build demographics records
        records = []
        for patient in patients:
            record = {
                "patient_id": patient.patient_id,
                "age": patient.age,
                "gender": patient.gender,
                "bmi": patient.bmi,
                "height_cm": patient.height_cm,
                "weight_kg": patient.weight_kg,
                "resting_hr_baseline": patient.resting_hr_baseline,
                "max_heart_rate": patient.max_heart_rate,
                "insulin_sensitivity_factor": patient.insulin_sensitivity_factor,
            }
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        output_path = self.output_dir / "patient_demographics.parquet"
        
        # Handle append mode
        if append and output_path.exists():
            logger.info(f"Appending to existing demographics file: {output_path}")
            print(f"  Appending to existing file...")
            
            # Read existing data
            existing_df = pd.read_parquet(output_path)
            
            # Combine with new data (remove duplicates based on patient_id)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['patient_id'], keep='last')
            
            # Write combined data
            combined_df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
            
            logger.info(f"Appended {len(df)} new patients, total now: {len(combined_df)}")
        else:
            # Write new file
            df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
            logger.info(f"Created new demographics file with {len(df)} patients")
        
        final_size = output_path.stat().st_size / 1024 / 1024
        print(f"✓ Demographics table written: {output_path}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  File size: {final_size:.2f} MB")
        
        logger.info(f"Demographics file size: {final_size:.2f} MB")
        
        return output_path
    
    def write_timeseries_table(
        self,
        windows: List[TimeSeriesWindow],
        labels: List[Dict[str, int]],
        window_size: int,
        split_name: str,
        append: bool = True,
    ) -> Path:
        """Write time-series features table to Parquet with append support.
        
        Args:
            windows: List of time series windows
            labels: List of label dictionaries
            window_size: Number of timesteps in window
            split_name: Name of split (train/val/test)
            append: If True, append to existing file if it exists
            
        Returns:
            Path to written file
        """
        logger.info(f"Writing time-series table for {split_name} split: {len(windows):,} windows (append={append})")
        
        print(f"\n{'='*60}")
        print(f"Writing Time-Series Table: {split_name.upper()}")
        print(f"{'='*60}")
        
        records = []
        
        for i, (window, label_dict) in enumerate(zip(windows, labels)):
            record = {
                "patient_id": window.patient_id,
                "timestamp": window.timestamp,
            }
            
            # Add time-series features (flattened)
            # Format: feature_t0, feature_t1, ..., feature_t4
            for feature_name, values in window.features.items():
                for t in range(window_size):
                    col_name = f"feature_t{t}_{feature_name}"
                    record[col_name] = values[t]
            
            # Add regression targets
            for target_name, target_value in window.target_values.items():
                record[f"target_{target_name}"] = target_value
            
            # Add classification labels
            record.update(label_dict)
            
            records.append(record)
            
            # Progress indicator
            if (i + 1) % 100000 == 0:
                msg = f"  Serialized: {i + 1:,}/{len(windows):,} windows..."
                print(msg)
                logger.info(msg)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        output_path = self.output_dir / f"timeseries_features_{split_name}.parquet"
        
        # Handle append mode
        if append and output_path.exists():
            logger.info(f"Appending to existing {split_name} file: {output_path}")
            print(f"  Appending to existing file...")
            
            # Read existing data
            existing_df = pd.read_parquet(output_path)
            
            # Combine with new data
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            # Write combined data
            combined_df.to_parquet(
                output_path,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            
            logger.info(f"Appended {len(df):,} windows to {split_name}, total now: {len(combined_df):,}")
            df = combined_df
        else:
            # Write new file
            df.to_parquet(
                output_path,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            logger.info(f"Created new {split_name} file with {len(df):,} windows")
        
        final_size = output_path.stat().st_size / 1024 / 1024
        print(f"✓ Time-series table written: {output_path}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  File size: {final_size:.2f} MB")
        
        logger.info(f"{split_name} file size: {final_size:.2f} MB, columns: {len(df.columns)}")
        
        return output_path
    
    def split_dataset(
        self,
        windows: List[TimeSeriesWindow],
        labels: List[Dict[str, int]],
        train_split: float,
        val_split: float,
        random_seed: int,
    ) -> tuple:
        """Split dataset into train/validation/test sets.
        
        Uses stratified split to maintain class balance.
        
        Args:
            windows: List of windows
            labels: List of labels
            train_split: Fraction for training
            val_split: Fraction for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data) where each is (windows, labels)
        """
        logger.info(f"Splitting {len(windows):,} samples into train/val/test sets")
        
        print(f"\n{'='*60}")
        print("Splitting Dataset...")
        print(f"{'='*60}")
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create indices and shuffle
        indices = list(range(len(windows)))
        random.shuffle(indices)
        
        # Calculate split sizes
        n_total = len(indices)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        logger.info(f"Split sizes - Train: {n_train}, Val: {n_val}, Test: {len(test_indices)}")
        
        # Create splits
        train_windows = [windows[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        
        val_windows = [windows[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        test_windows = [windows[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        print(f"  Training set: {len(train_windows):,} samples ({len(train_windows)/n_total*100:.1f}%)")
        print(f"  Validation set: {len(val_windows):,} samples ({len(val_windows)/n_total*100:.1f}%)")
        print(f"  Test set: {len(test_windows):,} samples ({len(test_windows)/n_total*100:.1f}%)")
        
        logger.info(f"Dataset split complete")
        
        return (
            (train_windows, train_labels),
            (val_windows, val_labels),
            (test_windows, test_labels),
        )
    
    def write_metadata(
        self,
        config: object,
        num_patients: int,
        total_samples: int,
        train_samples: int,
        val_samples: int,
        test_samples: int,
        append: bool = True,
    ) -> None:
        """Write dataset metadata and summary with append support.
        
        Args:
            config: Dataset configuration object
            num_patients: Number of patients
            total_samples: Total number of samples
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
            append: If True, append to existing file if it exists
        """
        logger.info(f"Writing metadata (append={append})")
        
        metadata_path = self.output_dir / "dataset_metadata.txt"
        
        mode = "a" if (append and metadata_path.exists()) else "w"
        
        with open(metadata_path, mode) as f:
            if mode == "a":
                f.write("\n\n" + "=" * 70 + "\n")
                f.write("ADDITIONAL GENERATION RUN\n")
                f.write("=" * 70 + "\n\n")
            else:
                f.write("=" * 70 + "\n")
                f.write("ML DATASET METADATA\n")
                f.write("=" * 70 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Number of Patients: {num_patients:,}\n")
            f.write(f"  Simulation Days per Patient: {config.simulation_days_per_patient}\n")
            f.write(f"  Window Size: {config.window_size_minutes} timesteps\n")
            f.write(f"  Random Seed: {config.random_seed}\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write(f"  Total Samples: {total_samples:,}\n")
            f.write(f"  Training Samples: {train_samples:,} ({train_samples/total_samples*100:.1f}%)\n")
            f.write(f"  Validation Samples: {val_samples:,} ({val_samples/total_samples*100:.1f}%)\n")
            f.write(f"  Test Samples: {test_samples:,} ({test_samples/total_samples*100:.1f}%)\n\n")
            
            f.write("Features:\n")
            f.write(f"  Features per Timestep: {len(config.features_to_extract)}\n")
            f.write(f"  Total Input Features: {len(config.features_to_extract) * config.window_size_minutes}\n")
            for feature in config.features_to_extract:
                f.write(f"    - {feature}\n")
            
            f.write(f"\nRegression Targets: {len(config.regression_targets)}\n")
            for target in config.regression_targets:
                f.write(f"    - {target}\n")
            
            f.write(f"\nClassification Labels: 6\n")
            f.write("    - Hypoglycemia Risk\n")
            f.write("    - Hyperglycemia Risk\n")
            f.write("    - Fall Risk\n")
            f.write("    - Cardiac Anomaly\n")
            f.write("    - Severe Hypotension Risk\n")
            f.write("    - Autonomic Dysregulation\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Files Generated:\n")
            f.write("=" * 70 + "\n")
            f.write("  - patient_demographics.parquet\n")
            f.write("  - timeseries_features_train.parquet\n")
            f.write("  - timeseries_features_val.parquet\n")
            f.write("  - timeseries_features_test.parquet\n")
            f.write("  - dataset_metadata.txt\n")
        
        logger.info(f"Metadata file {'appended' if mode == 'a' else 'created'}: {metadata_path}")
        print(f"\n✓ Metadata written: {metadata_path}")

