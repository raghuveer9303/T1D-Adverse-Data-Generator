"""Enhanced metabolic modeling with improved insulin pharmacokinetics.

This module provides more clinically accurate modeling of:
- Separate rapid-acting and basal insulin pools
- Insulin action delays
- Dawn phenomenon
- Bolus insulin calculations
"""

from simulation.metabolic_enhanced.insulin_pools import (
    InsulinPools,
    evolve_insulin_pools,
)
from simulation.metabolic_enhanced.insulin_action import (
    InsulinActionBuffer,
    calculate_insulin_action,
)
from simulation.metabolic_enhanced.dawn_phenomenon import (
    calculate_dawn_glucose_rise,
    calculate_dawn_insulin_resistance,
)
from simulation.metabolic_enhanced.bolus_calculator import (
    calculate_bolus_insulin,
)
from simulation.metabolic_enhanced.enhanced_metabolic import (
    calculate_next_glucose_enhanced,
)

__all__ = [
    "InsulinPools",
    "evolve_insulin_pools",
    "InsulinActionBuffer",
    "calculate_insulin_action",
    "calculate_dawn_glucose_rise",
    "calculate_dawn_insulin_resistance",
    "calculate_bolus_insulin",
    "calculate_next_glucose_enhanced",
]

