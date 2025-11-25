"""Centralized configuration for the diabetic digital twin simulation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv:
    # Load environment variables from a local .env file when present.
    load_dotenv()


def _bool_from_env(var_name: str, default: bool = True) -> bool:
    """Interpret common truthy/falsey strings from the environment."""

    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration values controlling simulation runtime behavior."""

    population_size: int = int(os.getenv("SIM_POPULATION_SIZE", "5"))
    chunk_size: int = int(os.getenv("SIM_CHUNK_SIZE", "250"))
    state_snapshot_path: Path = Path(
        os.getenv("SIM_STATE_SNAPSHOT_PATH", "population_state.pkl")
    )


@dataclass(frozen=True)
class PubSubConfig:
    """Google Cloud Pub/Sub configuration."""

    enabled: bool = _bool_from_env("ENABLE_PUBSUB", default=True)
    project_id: Optional[str] = os.getenv("GCP_PROJECT_ID")
    topic_id: Optional[str] = os.getenv("PUBSUB_TOPIC_ID")


SIMULATION_CONFIG = SimulationConfig()
PUBSUB_CONFIG = PubSubConfig()

