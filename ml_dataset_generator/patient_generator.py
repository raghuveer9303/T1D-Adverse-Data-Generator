"""Generate diverse patient profiles for ML dataset."""

from typing import List

import numpy as np

from data_models import PatientStaticProfile


class PatientGenerator:
    """Generate diverse patient profiles with controlled demographics.
    
    Single Responsibility: Generate patient profiles only.
    This class ensures demographic diversity across age, BMI, and gender.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize patient generator.
        
        Args:
            random_seed: Seed for reproducible patient generation
        """
        self.rng = np.random.default_rng(random_seed)
    
    def generate_patients(self, num_patients: int) -> List[PatientStaticProfile]:
        """Generate a diverse population of patients.
        
        Ensures demographic diversity by using existing patient generation logic.
        
        Args:
            num_patients: Number of patients to generate
            
        Returns:
            List of PatientStaticProfile objects
        """
        patients = []
        
        print(f"\n{'='*60}")
        print(f"Generating {num_patients:,} diverse patient profiles...")
        print(f"{'='*60}")
        
        for i in range(num_patients):
            # Generate patient using existing cascading demographics
            patient = PatientStaticProfile.generate_random(rng=self.rng)
            patients.append(patient)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1:,}/{num_patients:,} patients ({(i+1)/num_patients*100:.1f}%)")
        
        # Report demographic statistics
        self._report_demographics(patients)
        
        return patients
    
    def _report_demographics(self, patients: List[PatientStaticProfile]) -> None:
        """Print demographic statistics of generated patients.
        
        Args:
            patients: List of generated patient profiles
        """
        ages = [p.age for p in patients]
        bmis = [p.bmi for p in patients]
        genders = [p.gender for p in patients]
        sensitivities = [p.insulin_sensitivity_factor for p in patients]
        
        print(f"\n{'='*60}")
        print("Patient Demographics Summary")
        print(f"{'='*60}")
        
        print(f"\nAge Statistics:")
        print(f"  Mean: {np.mean(ages):.1f} ± {np.std(ages):.1f} years")
        print(f"  Range: {min(ages)} - {max(ages)} years")
        print(f"  Median: {np.median(ages):.1f} years")
        
        print(f"\nBMI Statistics:")
        print(f"  Mean: {np.mean(bmis):.1f} ± {np.std(bmis):.1f}")
        print(f"  Range: {min(bmis):.1f} - {max(bmis):.1f}")
        print(f"  Median: {np.median(bmis):.1f}")
        
        print(f"\nGender Distribution:")
        gender_counts = {"M": genders.count("M"), "F": genders.count("F")}
        print(f"  Male: {gender_counts['M']:,} ({gender_counts['M']/len(patients)*100:.1f}%)")
        print(f"  Female: {gender_counts['F']:,} ({gender_counts['F']/len(patients)*100:.1f}%)")
        
        print(f"\nBMI Categories:")
        bmi_categories = {
            "Underweight (<18.5)": sum(1 for b in bmis if b < 18.5),
            "Normal (18.5-25)": sum(1 for b in bmis if 18.5 <= b < 25),
            "Overweight (25-30)": sum(1 for b in bmis if 25 <= b < 30),
            "Obese (≥30)": sum(1 for b in bmis if b >= 30),
        }
        for category, count in bmi_categories.items():
            print(f"  {category}: {count:,} ({count/len(patients)*100:.1f}%)")
        
        print(f"\nAge Groups:")
        age_groups = {
            "Young Adults (18-35)": sum(1 for a in ages if 18 <= a < 35),
            "Middle-Aged (35-60)": sum(1 for a in ages if 35 <= a < 60),
            "Elderly (≥60)": sum(1 for a in ages if a >= 60),
        }
        for group, count in age_groups.items():
            print(f"  {group}: {count:,} ({count/len(patients)*100:.1f}%)")
        
        print(f"\nInsulin Sensitivity:")
        print(f"  Mean: {np.mean(sensitivities):.3f} ± {np.std(sensitivities):.3f}")
        print(f"  Range: {min(sensitivities):.3f} - {max(sensitivities):.3f}")
        print(f"  Resistant (<0.7): {sum(1 for s in sensitivities if s < 0.7):,} "
              f"({sum(1 for s in sensitivities if s < 0.7)/len(patients)*100:.1f}%)")
        
        print(f"{'='*60}\n")

