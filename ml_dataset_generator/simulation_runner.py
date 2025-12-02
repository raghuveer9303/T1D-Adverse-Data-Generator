"""Run simulations to collect time-series data."""

import logging
import multiprocessing as mp
from datetime import datetime
from functools import partial
from typing import List, Tuple

import numpy as np

from data_models import (
    ActivityMode,
    PatientDynamicState,
    PatientStaticProfile,
    SensorPayload,
)
from simulation_engine import process_single_patient

# Configure logging
logger = logging.getLogger(__name__)


class SimulationRunner:
    """Run patient simulations to generate time-series data.
    
    Single Responsibility: Execute simulations and collect sensor payloads.
    Open/Closed Principle: Can be extended without modification.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize simulation runner.
        
        Args:
            random_seed: Base random seed for simulations
        """
        self.base_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
    
    def run_patient_simulation(
        self,
        patient: PatientStaticProfile,
        num_days: int,
        patient_index: int,
    ) -> List[SensorPayload]:
        """Run a multi-day simulation for a single patient.
        
        Args:
            patient: Patient profile to simulate
            num_days: Number of days to simulate
            patient_index: Index of patient (for seeding)
            
        Returns:
            List of SensorPayload objects (one per minute)
        """
        # Create patient-specific RNG for reproducibility
        patient_rng = np.random.default_rng(self.base_seed + patient_index)
        
        # Initialize patient state at start time (midnight)
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        state = PatientDynamicState(
            timestamp_utc=start_time,
            simulation_tick=0,
            current_activity_mode=ActivityMode.SLEEP,
            activity_intensity=0.0,
            cumulative_fatigue=0.0,
        )
        
        # Calculate total minutes to simulate
        total_minutes = num_days * 1440
        
        # Collect payloads
        payloads = []
        
        for minute in range(total_minutes):
            # Process one minute forward
            state, payload = process_single_patient(
                profile=patient,
                state=state,
                rng=patient_rng,
            )
            
            payloads.append(payload)
        
        return payloads
    
    def run_batch_simulations(
        self,
        patients: List[PatientStaticProfile],
        num_days: int,
        start_index: int = 0,
        batch_num: int = 1,
        total_batches: int = 1,
        use_parallel: bool = True,
        n_jobs: int = None,
    ) -> List[Tuple[PatientStaticProfile, List[SensorPayload]]]:
        """Run simulations for a batch of patients using parallel processing.
        
        Args:
            patients: List of patient profiles
            num_days: Number of days to simulate per patient
            start_index: Starting index for patient seeding
            batch_num: Current batch number (for progress reporting)
            total_batches: Total number of batches
            use_parallel: Whether to use parallel processing (default: True)
            n_jobs: Number of parallel jobs. If None, uses all CPU cores
            
        Returns:
            List of tuples (patient, payloads)
        """
        logger.info(f"Starting Batch {batch_num}/{total_batches}")
        logger.info(f"Simulating {len(patients)} patients for {num_days} days each")
        
        print(f"\n{'='*60}")
        print(f"Running Batch {batch_num}/{total_batches}")
        print(f"Simulating {len(patients)} patients for {num_days} days each...")
        
        if use_parallel:
            if n_jobs is None:
                n_jobs = mp.cpu_count()
            print(f"Using {n_jobs} CPU cores for parallel processing")
            logger.info(f"Parallel processing enabled with {n_jobs} workers")
        else:
            print(f"Using sequential processing")
            logger.info(f"Sequential processing mode")
        
        print(f"{'='*60}")
        
        if use_parallel and len(patients) > 1:
            # Use multiprocessing for parallel execution
            results = self._run_parallel(patients, num_days, start_index, n_jobs)
        else:
            # Sequential execution
            results = self._run_sequential(patients, num_days, start_index)
        
        logger.info(f"Batch {batch_num} complete! Generated {len(results)} patient simulations")
        print(f"✓ Batch {batch_num} complete!")
        
        return results
    
    def _run_sequential(
        self,
        patients: List[PatientStaticProfile],
        num_days: int,
        start_index: int,
    ) -> List[Tuple[PatientStaticProfile, List[SensorPayload]]]:
        """Run simulations sequentially (non-parallel).
        
        Args:
            patients: List of patient profiles
            num_days: Number of days to simulate
            start_index: Starting index for patient seeding
            
        Returns:
            List of (patient, payloads) tuples
        """
        results = []
        
        for i, patient in enumerate(patients):
            patient_idx = start_index + i
            
            # Run simulation
            payloads = self.run_patient_simulation(
                patient=patient,
                num_days=num_days,
                patient_index=patient_idx,
            )
            
            results.append((patient, payloads))
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(patients):
                minutes_per_patient = len(payloads)
                total_minutes = (i + 1) * minutes_per_patient
                msg = (f"  Completed: {i + 1}/{len(patients)} patients "
                       f"({(i+1)/len(patients)*100:.1f}%) - "
                       f"{total_minutes:,} total minutes generated")
                print(msg)
                logger.info(msg)
        
        return results
    
    def _run_parallel(
        self,
        patients: List[PatientStaticProfile],
        num_days: int,
        start_index: int,
        n_jobs: int,
    ) -> List[Tuple[PatientStaticProfile, List[SensorPayload]]]:
        """Run simulations in parallel using multiprocessing.
        
        Args:
            patients: List of patient profiles
            num_days: Number of days to simulate
            start_index: Starting index for patient seeding
            n_jobs: Number of parallel workers
            
        Returns:
            List of (patient, payloads) tuples
        """
        # Create argument tuples for each patient
        args_list = [
            (patient, num_days, start_index + i)
            for i, patient in enumerate(patients)
        ]
        
        # Use multiprocessing pool
        with mp.Pool(processes=n_jobs) as pool:
            # Use imap_unordered for non-blocking parallel execution
            results = []
            completed = 0
            
            # Process results as they complete
            for result in pool.imap_unordered(_simulate_patient_wrapper, args_list):
                results.append(result)
                completed += 1
                
                # Progress indicator
                if completed % 10 == 0 or completed == len(patients):
                    minutes_per_patient = len(result[1])
                    total_minutes = completed * minutes_per_patient
                    msg = (f"  Completed: {completed}/{len(patients)} patients "
                           f"({completed/len(patients)*100:.1f}%) - "
                           f"{total_minutes:,} total minutes generated")
                    print(msg)
                    logger.info(msg)
        
        # Sort results by patient_id to maintain deterministic order
        results.sort(key=lambda x: x[0].patient_id)
        
        return results


def _simulate_patient_wrapper(args: tuple) -> Tuple[PatientStaticProfile, List[SensorPayload]]:
    """Wrapper function for parallel simulation execution.
    
    This function is defined at module level to support multiprocessing pickling.
    
    Args:
        args: Tuple of (patient, num_days, patient_index)
        
    Returns:
        Tuple of (patient, payloads)
    """
    patient, num_days, patient_index = args
    
    # Create patient-specific RNG
    patient_rng = np.random.default_rng(42 + patient_index)
    
    # Initialize patient state
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    state = PatientDynamicState(
        timestamp_utc=start_time,
        simulation_tick=0,
        current_activity_mode=ActivityMode.SLEEP,
        activity_intensity=0.0,
        cumulative_fatigue=0.0,
    )
    
    # Calculate total minutes
    total_minutes = num_days * 1440
    
    # Collect payloads
    payloads = []
    for minute in range(total_minutes):
        state, payload = process_single_patient(
            profile=patient,
            state=state,
            rng=patient_rng,
        )
        payloads.append(payload)
    
    return (patient, payloads)

