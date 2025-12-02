"""Orchestrate the entire ML dataset generation pipeline."""

import logging
import multiprocessing as mp
from typing import List

from data_models import PatientStaticProfile
from ml_dataset_generator.dataset_config import DatasetConfig
from ml_dataset_generator.patient_generator import PatientGenerator
from ml_dataset_generator.simulation_runner import SimulationRunner
from ml_dataset_generator.window_extractor import WindowExtractor
from ml_dataset_generator.label_calculator import LabelCalculator
from ml_dataset_generator.dataset_writer import DatasetWriter

# Configure logging
logger = logging.getLogger(__name__)


class MLDatasetOrchestrator:
    """Orchestrate the ML dataset generation pipeline.
    
    Follows Open/Closed Principle: Extensible without modification.
    Follows Dependency Inversion Principle: Depends on abstractions (components).
    """
    
    def __init__(self, config: DatasetConfig):
        """Initialize orchestrator with configuration.
        
        Args:
            config: Dataset generation configuration
        """
        self.config = config
        
        # Initialize components
        self.patient_generator = PatientGenerator(random_seed=config.random_seed)
        self.simulation_runner = SimulationRunner(random_seed=config.random_seed)
        self.window_extractor = WindowExtractor(
            window_size=config.window_size_minutes,
            features_to_extract=config.features_to_extract,
        )
        self.label_calculator = LabelCalculator()
        self.dataset_writer = DatasetWriter(output_dir=config.output_dir)
    
    def generate_full_dataset(
        self,
        use_parallel: bool = True,
        n_jobs: int = None,
        append_mode: bool = True,
    ) -> None:
        """Generate the complete ML dataset with parallel processing support.
        
        Orchestrates the entire pipeline from patient generation to file output.
        
        Args:
            use_parallel: Whether to use parallel processing (default: True)
            n_jobs: Number of parallel jobs. If None, uses all CPU cores
            append_mode: Whether to append to existing files (default: True)
        """
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        
        logger.info("="*70)
        logger.info("Starting ML Dataset Generation Pipeline")
        logger.info(f"Parallel processing: {use_parallel}, Workers: {n_jobs if use_parallel else 'N/A'}")
        logger.info(f"Append mode: {append_mode}")
        logger.info("="*70)
        
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 15 + "ML DATASET GENERATION PIPELINE" + " " * 22 + "║")
        print("╚" + "═" * 68 + "╝")
        
        if use_parallel:
            print(f"\n🚀 Parallel Processing: ENABLED ({n_jobs} CPU cores)")
        else:
            print(f"\n⚙️  Parallel Processing: DISABLED (Sequential mode)")
        
        print(f"📝 Append Mode: {'ENABLED' if append_mode else 'DISABLED'}")
        
        print(self.config.summary())
        
        # Step 1: Generate patients
        print("\n" + "─" * 70)
        print("STEP 1: Generate Patient Profiles")
        print("─" * 70)
        logger.info("Step 1: Generating patient profiles")
        patients = self.patient_generator.generate_patients(
            num_patients=self.config.num_patients
        )
        logger.info(f"Generated {len(patients)} patient profiles")
        
        # Step 2: Run simulations in batches
        print("\n" + "─" * 70)
        print("STEP 2: Run Patient Simulations")
        print("─" * 70)
        logger.info("Step 2: Running patient simulations")
        all_patient_payloads = self._run_simulations_batched(
            patients, use_parallel, n_jobs
        )
        logger.info(f"Completed {len(all_patient_payloads)} patient simulations")
        
        # Step 3: Extract windows
        print("\n" + "─" * 70)
        print("STEP 3: Extract Time-Series Windows")
        print("─" * 70)
        logger.info("Step 3: Extracting time-series windows")
        all_windows = self.window_extractor.extract_batch_windows(
            patient_payload_pairs=all_patient_payloads,
            regression_targets=self.config.regression_targets,
            use_parallel=use_parallel,
            n_jobs=n_jobs,
        )
        logger.info(f"Extracted {len(all_windows):,} time-series windows")
        
        # Step 4: Calculate labels
        print("\n" + "─" * 70)
        print("STEP 4: Calculate Adverse Event Labels")
        print("─" * 70)
        logger.info("Step 4: Calculating adverse event labels")
        all_labels = self.label_calculator.calculate_all_labels(
            all_windows,
            use_parallel=use_parallel,
            n_jobs=n_jobs,
        )
        logger.info(f"Calculated labels for {len(all_labels):,} windows")
        
        # Step 5: Split dataset
        print("\n" + "─" * 70)
        print("STEP 5: Split Dataset (Train/Val/Test)")
        print("─" * 70)
        logger.info("Step 5: Splitting dataset into train/val/test")
        train_data, val_data, test_data = self.dataset_writer.split_dataset(
            windows=all_windows,
            labels=all_labels,
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            random_seed=self.config.random_seed,
        )
        
        # Step 6: Write to Parquet files
        print("\n" + "─" * 70)
        print("STEP 6: Write Dataset Files")
        print("─" * 70)
        logger.info("Step 6: Writing dataset files to Parquet format")
        
        # Write demographics (shared across all splits)
        self.dataset_writer.write_demographics_table(patients, append=append_mode)
        
        # Write time-series tables for each split
        train_windows, train_labels = train_data
        val_windows, val_labels = val_data
        test_windows, test_labels = test_data
        
        self.dataset_writer.write_timeseries_table(
            windows=train_windows,
            labels=train_labels,
            window_size=self.config.window_size_minutes,
            split_name="train",
            append=append_mode,
        )
        
        self.dataset_writer.write_timeseries_table(
            windows=val_windows,
            labels=val_labels,
            window_size=self.config.window_size_minutes,
            split_name="val",
            append=append_mode,
        )
        
        self.dataset_writer.write_timeseries_table(
            windows=test_windows,
            labels=test_labels,
            window_size=self.config.window_size_minutes,
            split_name="test",
            append=append_mode,
        )
        
        # Write metadata
        self.dataset_writer.write_metadata(
            config=self.config,
            num_patients=len(patients),
            total_samples=len(all_windows),
            train_samples=len(train_windows),
            val_samples=len(val_windows),
            test_samples=len(test_windows),
            append=append_mode,
        )
        
        logger.info("All dataset files written successfully")
        
        # Final summary
        self._print_final_summary(
            num_patients=len(patients),
            total_samples=len(all_windows),
            train_samples=len(train_windows),
            val_samples=len(val_windows),
            test_samples=len(test_windows),
        )
    
    def _run_simulations_batched(
        self,
        patients: List[PatientStaticProfile],
        use_parallel: bool = True,
        n_jobs: int = None,
    ) -> List[tuple]:
        """Run simulations in batches to manage memory.
        
        Args:
            patients: List of all patients
            use_parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs
            
        Returns:
            List of (patient, payloads) tuples
        """
        all_results = []
        num_batches = (len(patients) + self.config.batch_size - 1) // self.config.batch_size
        
        logger.info(f"Running {num_batches} batches with batch_size={self.config.batch_size}")
        
        for batch_num in range(num_batches):
            start_idx = batch_num * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(patients))
            batch_patients = patients[start_idx:end_idx]
            
            batch_results = self.simulation_runner.run_batch_simulations(
                patients=batch_patients,
                num_days=self.config.simulation_days_per_patient,
                start_index=start_idx,
                batch_num=batch_num + 1,
                total_batches=num_batches,
                use_parallel=use_parallel,
                n_jobs=n_jobs,
            )
            
            all_results.extend(batch_results)
        
        return all_results
    
    def _print_final_summary(
        self,
        num_patients: int,
        total_samples: int,
        train_samples: int,
        val_samples: int,
        test_samples: int,
    ) -> None:
        """Print final generation summary.
        
        Args:
            num_patients: Number of patients generated
            total_samples: Total samples generated
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
        """
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "GENERATION COMPLETE!" + " " * 27 + "║")
        print("╚" + "═" * 68 + "╝")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  ✓ Patients: {num_patients:,}")
        print(f"  ✓ Total Samples: {total_samples:,}")
        print(f"  ✓ Training: {train_samples:,}")
        print(f"  ✓ Validation: {val_samples:,}")
        print(f"  ✓ Test: {test_samples:,}")
        
        print(f"\n📁 Output Location:")
        print(f"  {self.config.output_dir.absolute()}")
        
        print(f"\n📄 Files Generated:")
        print(f"  ✓ patient_demographics.parquet")
        print(f"  ✓ timeseries_features_train.parquet")
        print(f"  ✓ timeseries_features_val.parquet")
        print(f"  ✓ timeseries_features_test.parquet")
        print(f"  ✓ dataset_metadata.txt")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Load data in Spark:")
        print(f"     demographics = spark.read.parquet('patient_demographics.parquet')")
        print(f"     train = spark.read.parquet('timeseries_features_train.parquet')")
        print(f"  2. Join tables:")
        print(f"     dataset = train.join(demographics, on='patient_id')")
        print(f"  3. Train models with Spark MLlib")
        
        print(f"\n{'='*70}\n")

