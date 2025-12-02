"""Calculate classification labels for adverse events."""

import logging
import multiprocessing as mp
from typing import Dict, List

import numpy as np

from ml_dataset_generator.window_extractor import TimeSeriesWindow

# Configure logging
logger = logging.getLogger(__name__)


class LabelCalculator:
    """Calculate adverse event labels based on physiological rules.
    
    Single Responsibility: Calculate classification labels only.
    Each label has a clear medical/physiological justification.
    """
    
    def calculate_hypoglycemia_risk(self, window: TimeSeriesWindow) -> int:
        """Detect hypoglycemia risk (low blood glucose).
        
        Rule: glucose < 80 mg/dL AND trending down (delta < -3 mg/dL/min)
        OR severe hypoglycemia: glucose < 70 mg/dL
        
        Args:
            window: Time series window
            
        Returns:
            1 if hypoglycemia risk detected, 0 otherwise
        """
        glucose_values = window.features["glucose_mgdl"]
        current_glucose = glucose_values[-1]  # t=0
        
        # Severe hypoglycemia
        if current_glucose < 70.0:
            return 1
        
        # Moderate hypoglycemia with downward trend
        if current_glucose < 80.0 and len(glucose_values) >= 2:
            prev_glucose = glucose_values[-2]  # t=-1
            delta = current_glucose - prev_glucose
            if delta < -3.0:
                return 1
        
        return 0
    
    def calculate_hyperglycemia_risk(self, window: TimeSeriesWindow) -> int:
        """Detect hyperglycemia risk (high blood glucose).
        
        Rule: glucose > 180 mg/dL AND trending up (delta > 5 mg/dL/min)
        OR severe hyperglycemia: glucose > 250 mg/dL
        
        Args:
            window: Time series window
            
        Returns:
            1 if hyperglycemia risk detected, 0 otherwise
        """
        glucose_values = window.features["glucose_mgdl"]
        current_glucose = glucose_values[-1]
        
        # Severe hyperglycemia
        if current_glucose > 250.0:
            return 1
        
        # Moderate hyperglycemia with upward trend
        if current_glucose > 180.0 and len(glucose_values) >= 2:
            prev_glucose = glucose_values[-2]
            delta = current_glucose - prev_glucose
            if delta > 5.0:
                return 1
        
        return 0
    
    def calculate_fall_risk(self, window: TimeSeriesWindow) -> int:
        """Detect fall risk from combined factors.
        
        Rule: (hypoglycemia OR glucose < 70) AND 
              (high activity > 0.5) AND 
              (low HRV < 20 ms)
        
        Represents dizziness/coordination issues during activity.
        
        Args:
            window: Time series window
            
        Returns:
            1 if fall risk detected, 0 otherwise
        """
        glucose = window.features["glucose_mgdl"][-1]
        activity = window.features["activity_intensity"][-1]
        hrv = window.features["hrv_sdnn"][-1]
        
        # Hypoglycemia during activity with autonomic stress
        if glucose < 70.0 and activity > 0.5 and hrv < 20.0:
            return 1
        
        return 0
    
    def calculate_cardiac_anomaly(self, window: TimeSeriesWindow) -> int:
        """Detect cardiac anomalies.
        
        Rule: QTc > 470ms (prolonged, arrhythmia risk) OR
              heart_rate < 45 (severe bradycardia) OR
              heart_rate > 160 (severe tachycardia when not exercising)
        
        Args:
            window: Time series window
            
        Returns:
            1 if cardiac anomaly detected, 0 otherwise
        """
        heart_rate = window.features["heart_rate_bpm"][-1]
        qt_interval = window.features["qt_interval_ms"][-1]
        activity = window.features["activity_intensity"][-1]
        
        # Calculate QTc using Bazett's formula
        if heart_rate > 0:
            rr_seconds = 60.0 / heart_rate
            qtc = qt_interval / np.sqrt(rr_seconds)
            
            if qtc > 470.0:
                return 1
        
        # Severe bradycardia
        if heart_rate < 45.0:
            return 1
        
        # Severe tachycardia at rest/low activity
        if heart_rate > 160.0 and activity < 0.3:
            return 1
        
        return 0
    
    def calculate_severe_hypotension_risk(self, window: TimeSeriesWindow) -> int:
        """Detect severe hypotension risk.
        
        Rule: Very low glucose (< 60) + high activity + heart rate dropping
        Based on: HR_delta < -10 bpm AND glucose < 70 AND activity > 0.3
        
        Args:
            window: Time series window
            
        Returns:
            1 if severe hypotension risk detected, 0 otherwise
        """
        glucose = window.features["glucose_mgdl"][-1]
        activity = window.features["activity_intensity"][-1]
        hr_values = window.features["heart_rate_bpm"]
        
        if len(hr_values) >= 2:
            current_hr = hr_values[-1]
            prev_hr = hr_values[-2]
            hr_delta = current_hr - prev_hr
            
            # Glucose crash with dropping HR during activity
            if glucose < 70.0 and activity > 0.3 and hr_delta < -10.0:
                return 1
        
        return 0
    
    def calculate_autonomic_dysregulation(self, window: TimeSeriesWindow) -> int:
        """Detect autonomic nervous system dysregulation.
        
        Rule: High EDA (> 12 μS) + abnormal HRV (< 15 ms) + glucose extremes
        Represents autonomic stress from metabolic dysregulation.
        
        Args:
            window: Time series window
            
        Returns:
            1 if autonomic dysregulation detected, 0 otherwise
        """
        eda = window.features["eda_microsiemens"][-1]
        hrv = window.features["hrv_sdnn"][-1]
        glucose = window.features["glucose_mgdl"][-1]
        
        # High sympathetic activity with low parasympathetic tone
        if eda > 12.0 and hrv < 15.0:
            # During glucose extremes
            if glucose < 70.0 or glucose > 200.0:
                return 1
        
        return 0
    
    def calculate_all_labels(
        self,
        windows: List[TimeSeriesWindow],
        use_parallel: bool = True,
        n_jobs: int = None,
        chunk_size: int = 1000,
    ) -> List[Dict[str, int]]:
        """Calculate all classification labels for a list of windows using parallel processing.
        
        Args:
            windows: List of time series windows
            use_parallel: Whether to use parallel processing (default: True)
            n_jobs: Number of parallel jobs. If None, uses all CPU cores
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            List of label dictionaries
        """
        logger.info(f"Starting label calculation for {len(windows):,} windows")
        
        print(f"\n{'='*60}")
        print(f"Calculating adverse event labels...")
        
        if use_parallel:
            if n_jobs is None:
                n_jobs = mp.cpu_count()
            print(f"Using {n_jobs} CPU cores for parallel processing")
            logger.info(f"Parallel processing enabled with {n_jobs} workers")
        else:
            print(f"Using sequential processing")
            logger.info(f"Sequential processing mode")
        
        print(f"{'='*60}")
        
        if use_parallel and len(windows) > chunk_size:
            all_labels = self._calculate_parallel(windows, n_jobs, chunk_size)
        else:
            all_labels = self._calculate_sequential(windows)
        
        # Calculate label statistics
        self._print_label_statistics(all_labels)
        
        logger.info(f"Label calculation complete for {len(all_labels):,} windows")
        
        return all_labels
    
    def _calculate_sequential(
        self,
        windows: List[TimeSeriesWindow],
    ) -> List[Dict[str, int]]:
        """Calculate labels sequentially (non-parallel).
        
        Args:
            windows: List of time series windows
            
        Returns:
            List of label dictionaries
        """
        all_labels = []
        
        for i, window in enumerate(windows):
            labels = {
                "label_hypoglycemia_risk": self.calculate_hypoglycemia_risk(window),
                "label_hyperglycemia_risk": self.calculate_hyperglycemia_risk(window),
                "label_fall_risk": self.calculate_fall_risk(window),
                "label_cardiac_anomaly": self.calculate_cardiac_anomaly(window),
                "label_severe_hypotension_risk": self.calculate_severe_hypotension_risk(window),
                "label_autonomic_dysregulation": self.calculate_autonomic_dysregulation(window),
            }
            
            all_labels.append(labels)
            
            if (i + 1) % 10000 == 0 or (i + 1) == len(windows):
                msg = (f"  Processed: {i + 1:,}/{len(windows):,} windows "
                       f"({(i+1)/len(windows)*100:.1f}%)")
                print(msg)
                logger.info(msg)
        
        return all_labels
    
    def _calculate_parallel(
        self,
        windows: List[TimeSeriesWindow],
        n_jobs: int,
        chunk_size: int,
    ) -> List[Dict[str, int]]:
        """Calculate labels in parallel using multiprocessing.
        
        Args:
            windows: List of time series windows
            n_jobs: Number of parallel workers
            chunk_size: Size of chunks to process
            
        Returns:
            List of label dictionaries
        """
        # Split windows into chunks for better load balancing
        chunks = [windows[i:i + chunk_size] for i in range(0, len(windows), chunk_size)]
        
        logger.info(f"Split {len(windows):,} windows into {len(chunks)} chunks of size {chunk_size}")
        
        all_labels = []
        completed_chunks = 0
        total_processed = 0
        
        # Use multiprocessing pool
        with mp.Pool(processes=n_jobs) as pool:
            # Process chunks as they complete (non-blocking)
            for chunk_labels in pool.imap_unordered(_calculate_labels_chunk, chunks):
                all_labels.extend(chunk_labels)
                completed_chunks += 1
                total_processed += len(chunk_labels)
                
                # Progress indicator
                msg = (f"  Processed: {total_processed:,}/{len(windows):,} windows "
                       f"({total_processed/len(windows)*100:.1f}%) - "
                       f"{completed_chunks}/{len(chunks)} chunks complete")
                print(msg)
                logger.info(msg)
        
        return all_labels


def _calculate_labels_chunk(windows_chunk: List[TimeSeriesWindow]) -> List[Dict[str, int]]:
    """Calculate labels for a chunk of windows.
    
    This function is defined at module level to support multiprocessing pickling.
    
    Args:
        windows_chunk: Chunk of TimeSeriesWindow objects
        
    Returns:
        List of label dictionaries for the chunk
    """
    calculator = LabelCalculator()
    
    labels_chunk = []
    for window in windows_chunk:
        labels = {
            "label_hypoglycemia_risk": calculator.calculate_hypoglycemia_risk(window),
            "label_hyperglycemia_risk": calculator.calculate_hyperglycemia_risk(window),
            "label_fall_risk": calculator.calculate_fall_risk(window),
            "label_cardiac_anomaly": calculator.calculate_cardiac_anomaly(window),
            "label_severe_hypotension_risk": calculator.calculate_severe_hypotension_risk(window),
            "label_autonomic_dysregulation": calculator.calculate_autonomic_dysregulation(window),
        }
        labels_chunk.append(labels)
    
    return labels_chunk
    
    def _print_label_statistics(self, all_labels: List[Dict[str, int]]) -> None:
        """Print statistics about label distribution.
        
        Args:
            all_labels: List of label dictionaries
        """
        print(f"\n{'='*60}")
        print("Label Distribution Statistics")
        print(f"{'='*60}")
        
        total = len(all_labels)
        
        for label_name in all_labels[0].keys():
            positive_count = sum(labels[label_name] for labels in all_labels)
            percentage = (positive_count / total) * 100
            
            display_name = label_name.replace("label_", "").replace("_", " ").title()
            print(f"  {display_name}:")
            print(f"    Positive: {positive_count:,} ({percentage:.2f}%)")
            print(f"    Negative: {total - positive_count:,} ({100-percentage:.2f}%)")
        
        print(f"{'='*60}\n")

