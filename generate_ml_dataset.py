#!/usr/bin/env python3
"""Main script to generate ML dataset for Spark MLlib training.

This script generates a comprehensive time-series dataset with:
- Patient demographics table (static features)
- Time-series features table (sliding windows)
- Regression targets (glucose, heart rate, etc.)
- Classification labels (adverse event detection)

Usage:
    python generate_ml_dataset.py

The script will generate ~3 million samples from 1000 patients.
Output will be written to ./ml_dataset_output/ as Parquet files.

Features:
- Parallel processing using all CPU cores
- Console logging for progress tracking
- File appending support for incremental generation
"""

import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

from ml_dataset_generator.dataset_config import DatasetConfig
from ml_dataset_generator.orchestrator import MLDatasetOrchestrator


def setup_logging(output_dir: Path) -> None:
    """Configure logging to both console and file.
    
    Args:
        output_dir: Directory for log files
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"ml_dataset_generation_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler (append mode)
            logging.FileHandler(log_file, mode='a'),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Available CPU cores: {mp.cpu_count()}")
    logger.info("="*70)


def main():
    """Main entry point for ML dataset generation."""
    
    # Configure dataset generation
    config = DatasetConfig(
        # Patient configuration
        num_patients=1000,
        simulation_days_per_patient=3,
        
        # Time-series configuration
        window_size_minutes=5,
        
        # Output configuration
        output_dir=Path("ml_dataset_output"),
        train_split=0.70,
        val_split=0.15,
        # test_split is automatically 0.15
        
        # Performance tuning
        batch_size=50,  # Process 50 patients at a time
        random_seed=42,
    )
    
    # Setup logging
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    
    # Get number of CPU cores
    n_cores = mp.cpu_count()
    
    logger.info("Starting ML Dataset Generation")
    logger.info(f"Configuration: {config.num_patients} patients, {config.simulation_days_per_patient} days each")
    logger.info(f"Using all {n_cores} CPU cores for parallel processing")
    
    print(f"\n{'='*70}")
    print(f"🚀 ML DATASET GENERATOR")
    print(f"{'='*70}")
    print(f"📊 Patients: {config.num_patients:,}")
    print(f"📅 Days per patient: {config.simulation_days_per_patient}")
    print(f"⚡ CPU cores: {n_cores}")
    print(f"💾 Output: {config.output_dir.absolute()}")
    print(f"{'='*70}\n")
    
    # Create orchestrator and run pipeline
    orchestrator = MLDatasetOrchestrator(config)
    
    try:
        # Run with full parallel processing and append mode enabled
        orchestrator.generate_full_dataset(
            use_parallel=True,      # Use all CPU cores
            n_jobs=n_cores,         # Explicitly set number of workers
            append_mode=True,       # Enable file appending
        )
        
        logger.info("ML Dataset generation completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user!")
        print("\n\n⚠️  Generation interrupted by user!")
        print("Partial data may have been written to disk.")
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        print(f"\n\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

