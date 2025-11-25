"""Separate insulin pools for rapid-acting and basal insulin.

This module models the different pharmacokinetics of:
- Rapid-acting insulin (Lispro, Aspart): Fast decay, short duration
- Basal insulin (Glargine, Detemir): Slow decay, long duration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Basal insulin infusion rate (units/minute) - typical basal rate ~0.5-1.0 units/hour
BASAL_INSULIN_RATE = 0.015  # ~0.9 units/hour

# Rapid-acting insulin decay: 2% per minute (~35 min half-life)
RAPID_ACTING_DECAY_RATE = 0.98

# Basal insulin decay: 0.1% per minute (~24-hour half-life for long-acting)
BASAL_DECAY_RATE = 0.999


@dataclass
class InsulinPools:
    """Separate pools for rapid-acting (bolus) and basal insulin."""

    rapid_acting: float  # Bolus insulin (fast-acting, decays quickly)
    basal: float  # Basal insulin (long-acting, decays slowly)

    @classmethod
    def from_legacy(cls, insulin_on_board: float) -> InsulinPools:
        """Convert legacy single insulin value to separate pools.
        
        Assumes existing insulin is mostly rapid-acting (for backward compatibility).
        """
        return cls(
            rapid_acting=insulin_on_board * 0.9,  # 90% rapid-acting
            basal=insulin_on_board * 0.1,  # 10% basal
        )

    def to_legacy(self) -> float:
        """Convert to legacy single insulin value (for backward compatibility)."""
        return self.rapid_acting + self.basal

    def total(self) -> float:
        """Total insulin on board."""
        return self.rapid_acting + self.basal


def evolve_insulin_pools(
    pools: InsulinPools,
    add_basal: bool = True,
    add_bolus: float = 0.0,
) -> InsulinPools:
    """
    Evolve insulin pools by one minute.
    
    Args:
        pools: Current insulin pools
        add_basal: Whether to add continuous basal insulin
        add_bolus: Additional bolus insulin to add (units)
    
    Returns:
        Updated insulin pools after one minute of decay and infusion
    """
    # Rapid-acting insulin decays quickly
    new_rapid_acting = pools.rapid_acting * RAPID_ACTING_DECAY_RATE

    # Basal insulin decays slowly
    new_basal = pools.basal * BASAL_DECAY_RATE

    # Add continuous basal infusion (simulates pump or long-acting injection)
    if add_basal:
        new_basal += BASAL_INSULIN_RATE

    # Add bolus insulin (goes into rapid-acting pool)
    new_rapid_acting += add_bolus

    # Ensure non-negative
    new_rapid_acting = max(0.0, new_rapid_acting)
    new_basal = max(0.0, new_basal)

    return InsulinPools(
        rapid_acting=new_rapid_acting,
        basal=new_basal,
    )

