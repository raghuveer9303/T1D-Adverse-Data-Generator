"""Extract sliding windows from time-series data."""

import logging
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

from data_models import PatientStaticProfile, SensorPayload

# Configure logging
logger = logging.getLogger(__name__)


class TimeSeriesWindow:
    """Represents a single time-series window with features and metadata.
    
    Encapsulates window data following OOP principles.
    """
    
    def __init__(
        self,
        patient_id: str,
        timestamp: datetime,
        features: Dict[str, List[float]],
        target_values: Dict[str, float],
        raw_payloads: List[SensorPayload],
    ):
        """Initialize a time-series window.
        
        Args:
            patient_id: Patient identifier
            timestamp: Timestamp of the last point (t=0)
            features: Dictionary mapping feature names to lists of values
            target_values: Dictionary of target values at t+1
            raw_payloads: Raw sensor payloads for this window
        """
        self.patient_id = patient_id
        self.timestamp = timestamp
        self.features = features
        self.target_values = target_values
        self.raw_payloads = raw_payloads


class WindowExtractor:
    """Extract sliding windows from patient simulation data.
    
    Single Responsibility: Extract and structure time-series windows.
    """
    
    def __init__(self, window_size: int, features_to_extract: List[str]):
        """Initialize window extractor.
        
        Args:
            window_size: Number of timesteps in each window
            features_to_extract: List of feature names to extract
        """
        self.window_size = window_size
        self.features_to_extract = features_to_extract
    
    def extract_feature_value(
        self,
        payload: SensorPayload,
        feature_name: str,
    ) -> float:
        """Extract a single feature value from a sensor payload.
        
        Maps feature names to their locations in the payload structure.
        
        Args:
            payload: Sensor payload to extract from
            feature_name: Name of the feature to extract
            
        Returns:
            Float value of the feature
        """
        # Map feature names to payload locations
        if feature_name == "glucose_mgdl":
            return float(payload.metabolics["glucose_mgdl"])
        elif feature_name == "heart_rate_bpm":
            return float(payload.vitals["heart_rate_bpm"])
        elif feature_name == "hrv_sdnn":
            return float(payload.vitals["hrv_sdnn"])
        elif feature_name == "respiratory_rate_rpm":
            return float(payload.vitals["resp_rate_rpm"])
        elif feature_name == "spo2_pct":
            return float(payload.vitals["spo2_pct"])
        elif feature_name == "steps_per_minute":
            return float(payload.wearable["steps_per_minute"])
        elif feature_name == "vertical_acceleration_g":
            return float(payload.wearable["accel_y_g"])
        elif feature_name == "skin_temperature_c":
            return float(payload.wearable["skin_temp_c"])
        elif feature_name == "eda_microsiemens":
            return float(payload.wearable["eda_microsiemens"])
        elif feature_name == "qt_interval_ms":
            return float(payload.vitals["qt_interval_ms"])
        elif feature_name == "insulin_on_board":
            # Extract from meta/state if available, otherwise default
            return 0.5  # This would need to be passed from state
        elif feature_name == "carbs_in_stomach":
            return 0.0  # This would need to be passed from state
        elif feature_name == "activity_intensity":
            # Would need to be passed from state
            # Approximate from steps
            steps = payload.wearable["steps_per_minute"]
            if steps < 10:
                return 0.1
            elif steps < 60:
                return 0.3
            elif steps < 100:
                return 0.5
            elif steps < 140:
                return 0.7
            else:
                return 0.9
        elif feature_name == "hour_of_day":
            timestamp = payload.meta["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            return timestamp.hour + timestamp.minute / 60.0
        else:
            raise ValueError(f"Unknown feature: {feature_name}")
    
    def extract_windows(
        self,
        patient: PatientStaticProfile,
        payloads: List[SensorPayload],
        regression_targets: List[str],
    ) -> List[TimeSeriesWindow]:
        """Extract all possible windows from a patient's payloads.
        
        Uses sliding window approach with stride=1 minute.
        
        Args:
            patient: Patient profile
            payloads: List of sensor payloads (chronological order)
            regression_targets: List of target features to predict
            
        Returns:
            List of TimeSeriesWindow objects
        """
        windows = []
        
        # Need window_size + 1 payloads (window + target)
        min_length = self.window_size + 1
        
        if len(payloads) < min_length:
            return windows
        
        # Extract windows with sliding approach
        for i in range(len(payloads) - min_length + 1):
            window_payloads = payloads[i:i + self.window_size]
            target_payload = payloads[i + self.window_size]
            
            # Extract features for each timestep in window
            features_dict = {feature: [] for feature in self.features_to_extract}
            
            for payload in window_payloads:
                for feature in self.features_to_extract:
                    value = self.extract_feature_value(payload, feature)
                    features_dict[feature].append(value)
            
            # Extract target values
            target_values = {}
            for target_name in regression_targets:
                target_values[target_name] = self.extract_feature_value(
                    target_payload, target_name
                )
            
            # Get timestamp from last window payload (t=0)
            last_payload = window_payloads[-1]
            timestamp = last_payload.meta["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Create window object
            window = TimeSeriesWindow(
                patient_id=patient.patient_id,
                timestamp=timestamp,
                features=features_dict,
                target_values=target_values,
                raw_payloads=window_payloads + [target_payload],
            )
            
            windows.append(window)
        
        return windows
    
    def extract_batch_windows(
        self,
        patient_payload_pairs: List[tuple],
        regression_targets: List[str],
        use_parallel: bool = True,
        n_jobs: int = None,
    ) -> List[TimeSeriesWindow]:
        """Extract windows from multiple patients using parallel processing.
        
        Args:
            patient_payload_pairs: List of (patient, payloads) tuples
            regression_targets: List of target features
            use_parallel: Whether to use parallel processing (default: True)
            n_jobs: Number of parallel jobs. If None, uses all CPU cores
            
        Returns:
            List of all windows from all patients
        """
        logger.info(f"Starting window extraction for {len(patient_payload_pairs)} patients")
        
        print(f"\n{'='*60}")
        print(f"Extracting sliding windows...")
        
        if use_parallel:
            if n_jobs is None:
                n_jobs = mp.cpu_count()
            print(f"Using {n_jobs} CPU cores for parallel processing")
            logger.info(f"Parallel processing enabled with {n_jobs} workers")
        else:
            print(f"Using sequential processing")
            logger.info(f"Sequential processing mode")
        
        print(f"{'='*60}")
        
        if use_parallel and len(patient_payload_pairs) > 1:
            all_windows = self._extract_parallel(
                patient_payload_pairs, regression_targets, n_jobs
            )
        else:
            all_windows = self._extract_sequential(
                patient_payload_pairs, regression_targets
            )
        
        logger.info(f"Window extraction complete! Total windows: {len(all_windows):,}")
        print(f"✓ Window extraction complete! Total windows: {len(all_windows):,}")
        
        return all_windows
    
    def _extract_sequential(
        self,
        patient_payload_pairs: List[tuple],
        regression_targets: List[str],
    ) -> List[TimeSeriesWindow]:
        """Extract windows sequentially (non-parallel).
        
        Args:
            patient_payload_pairs: List of (patient, payloads) tuples
            regression_targets: List of target features
            
        Returns:
            List of all windows
        """
        all_windows = []
        
        for i, (patient, payloads) in enumerate(patient_payload_pairs):
            windows = self.extract_windows(patient, payloads, regression_targets)
            all_windows.extend(windows)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(patient_payload_pairs):
                msg = (f"  Processed: {i + 1}/{len(patient_payload_pairs)} patients - "
                       f"{len(all_windows):,} total windows extracted")
                print(msg)
                logger.info(msg)
        
        return all_windows
    
    def _extract_parallel(
        self,
        patient_payload_pairs: List[tuple],
        regression_targets: List[str],
        n_jobs: int,
    ) -> List[TimeSeriesWindow]:
        """Extract windows in parallel using multiprocessing.
        
        Args:
            patient_payload_pairs: List of (patient, payloads) tuples
            regression_targets: List of target features
            n_jobs: Number of parallel workers
            
        Returns:
            List of all windows
        """
        # Create wrapper arguments
        args_list = [
            (patient, payloads, regression_targets, self.window_size, self.features_to_extract)
            for patient, payloads in patient_payload_pairs
        ]
        
        all_windows = []
        completed = 0
        
        # Use multiprocessing pool
        with mp.Pool(processes=n_jobs) as pool:
            # Process results as they complete (non-blocking)
            for windows in pool.imap_unordered(_extract_windows_wrapper, args_list):
                all_windows.extend(windows)
                completed += 1
                
                # Progress indicator
                if completed % 10 == 0 or completed == len(patient_payload_pairs):
                    msg = (f"  Processed: {completed}/{len(patient_payload_pairs)} patients - "
                           f"{len(all_windows):,} total windows extracted")
                    print(msg)
                    logger.info(msg)
        
        return all_windows


def _extract_windows_wrapper(args: tuple) -> List[TimeSeriesWindow]:
    """Wrapper function for parallel window extraction.
    
    This function is defined at module level to support multiprocessing pickling.
    
    Args:
        args: Tuple of (patient, payloads, regression_targets, window_size, features_to_extract)
        
    Returns:
        List of TimeSeriesWindow objects
    """
    patient, payloads, regression_targets, window_size, features_to_extract = args
    
    # Create temporary extractor instance
    extractor = WindowExtractor(window_size, features_to_extract)
    
    # Extract windows for this patient
    windows = extractor.extract_windows(patient, payloads, regression_targets)
    
    return windows

