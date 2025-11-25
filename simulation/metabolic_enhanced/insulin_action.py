"""Insulin action delay modeling.

Models the delayed onset of insulin action after subcutaneous injection.
Rapid-acting insulin has a 15-30 minute delay before peak action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from simulation.metabolic_enhanced.insulin_pools import InsulinPools


# Insulin action delay parameters
ACTION_DELAY_MINUTES = 60  # Buffer size (1 hour)
PEAK_ACTION_DELAY = 45  # Peak action occurs ~45 minutes after injection


@dataclass
class InsulinActionBuffer:
    """Circular buffer tracking insulin action over time."""

    buffer: List[float]  # Insulin amounts at each minute delay
    current_index: int  # Current position in circular buffer

    @classmethod
    def create(cls, size: int = ACTION_DELAY_MINUTES) -> InsulinActionBuffer:
        """Create a new empty buffer."""
        return cls(
            buffer=[0.0] * size,
            current_index=0,
        )

    def add_insulin(self, amount: float) -> None:
        """Add insulin to the current time slot."""
        self.buffer[self.current_index] += amount

    def advance(self) -> None:
        """Advance time by one minute (circular buffer)."""
        # Move to next slot first
        self.current_index = (self.current_index + 1) % len(self.buffer)
        # Clear the new slot (it will be used for new insulin)
        self.buffer[self.current_index] = 0.0

    def get_current_action(self) -> float:
        """
        Calculate current insulin action using weighted sum.
        
        Peak action occurs at PEAK_ACTION_DELAY minutes ago.
        Uses a Gaussian-like weighting centered at the peak delay.
        """
        buffer_size = len(self.buffer)
        total_action = 0.0

        for i in range(buffer_size):
            # Calculate how many minutes ago this insulin was added
            minutes_ago = (self.current_index - i) % buffer_size
            if minutes_ago == 0:
                minutes_ago = buffer_size  # Handle wrap-around

            # Weight based on distance from peak action delay
            # Gaussian-like curve: peak at PEAK_ACTION_DELAY, sigma ~15 minutes
            distance_from_peak = abs(minutes_ago - PEAK_ACTION_DELAY)
            weight = np.exp(-(distance_from_peak ** 2) / (2 * 15.0 ** 2))

            total_action += self.buffer[i] * weight

        return float(total_action)


def calculate_insulin_action(
    buffer: InsulinActionBuffer,
    insulin_pools: InsulinPools,
    sensitivity_factor: float,
    new_bolus: float = 0.0,
) -> float:
    """
    Calculate effective insulin action considering delays.
    
    Args:
        buffer: Insulin action delay buffer
        insulin_pools: Current insulin pools
        sensitivity_factor: Patient's insulin sensitivity
        new_bolus: New bolus insulin to add to buffer (only new doses, not total pool)
    
    Returns:
        Effective insulin action (mg/dL reduction per minute)
    """
    # Only add NEW bolus insulin to buffer (not the entire pool repeatedly)
    # The buffer tracks individual insulin doses over time
    if new_bolus > 0.0:
        buffer.add_insulin(new_bolus)

    # Basal insulin acts immediately (it's already been infusing)
    # But with lower activity per unit (it's designed for steady state)
    basal_action = insulin_pools.basal * sensitivity_factor * 1.5

    # Get delayed action from buffered insulin doses
    delayed_action = buffer.get_current_action() * sensitivity_factor * 2.5

    # Advance buffer for next minute
    buffer.advance()

    return delayed_action + basal_action

