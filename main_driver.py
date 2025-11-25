"""Entry point for the diabetic digital twin swarm."""

from __future__ import annotations

import json
import logging
import pickle
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import List, Sequence, Tuple

import numpy as np

try:
    from google.cloud import pubsub_v1
except ImportError:  # pragma: no cover - optional dependency
    pubsub_v1 = None

from config import PUBSUB_CONFIG, SIMULATION_CONFIG, PubSubConfig
from data_models import ActivityMode, PatientDynamicState, PatientStaticProfile
from simulation_engine import evolve_patient_batch


class PubSubClient:
    """Thin wrapper around PublisherClient with batch optimizations."""

    def __init__(self, config: PubSubConfig):
        self._config = config
        self.publisher = None
        self.topic_path = None
        if not config.enabled:
            logging.info(
                "Pub/Sub disabled via config; payloads will be printed locally."
            )
            return

        if not pubsub_v1:
            logging.warning(
                "Pub/Sub dependency google-cloud-pubsub missing; "
                "payloads will be printed to stdout."
            )
            return

        project_id = config.project_id
        topic_id = config.topic_id
        if project_id and topic_id:
            batch_settings = pubsub_v1.types.BatchSettings(
                max_messages=100,
                max_latency=1.0,
            )
            self.publisher = pubsub_v1.PublisherClient(
                batch_settings=batch_settings
            )
            self.topic_path = self.publisher.topic_path(project_id, topic_id)
            logging.info(
                "Pub/Sub client initialized for topic %s", self.topic_path
            )
        else:
            logging.warning(
                "Pub/Sub configuration missing project/topic; "
                "payloads will be printed to stdout."
            )

    def publish(self, payload: dict) -> None:
        """Publish JSON payload or print when Pub/Sub unavailable."""

        message_bytes = json.dumps(payload).encode("utf-8")
        if self.publisher and self.topic_path:
            future = self.publisher.publish(self.topic_path, message_bytes)
            future.add_done_callback(lambda _: None)
        else:
            print(message_bytes.decode("utf-8"), file=sys.stdout)


def setup_population(
    population_size: int | None = None,
) -> Tuple[List[PatientStaticProfile], List[PatientDynamicState]]:
    """Bootstrap the cohort with stochastic physiology."""

    population_size = population_size or SIMULATION_CONFIG.population_size
    rng = np.random.default_rng()
    profiles: List[PatientStaticProfile] = []
    states: List[PatientDynamicState] = []
    insulin_resistant = 0

    start_time = datetime.now(timezone.utc)
    for _ in range(population_size):
        profile = PatientStaticProfile.generate_random(rng=rng)
        profiles.append(profile)
        if profile.insulin_sensitivity_factor < 1.0:
            insulin_resistant += 1

        state = PatientDynamicState(
            timestamp_utc=start_time,
            simulation_tick=0,
            current_activity_mode=ActivityMode.SLEEP,
            activity_intensity=0.0,
            cumulative_fatigue=0.0,
            meal_flags_date=start_time.date(),
        )
        states.append(state)

    ages = np.array([p.age for p in profiles], dtype=float)
    mean_age = ages.mean() if len(ages) else 0.0
    resistant_pct = (insulin_resistant / population_size) * 100.0

    logging.info(
        "Population created. Mean Age: %.1f, %% Insulin Resistant: %.1f%%",
        mean_age,
        resistant_pct,
    )

    return profiles, states


def _chunk_indices(total: int, chunk_size: int) -> List[range]:
    """Precompute contiguous index windows for the executor."""
    return [
        range(start, min(start + chunk_size, total))
        for start in range(0, total, chunk_size)
    ]
    
def run_simulation_loop(
    profiles: Sequence[PatientStaticProfile],
    states: List[PatientDynamicState],
    pubsub_client: PubSubClient,
    chunk_size: int | None = None,
) -> None:
    """Main scheduling loop with drift correction and graceful shutdown."""

    chunk_size = chunk_size or SIMULATION_CONFIG.chunk_size
    chunk_windows = _chunk_indices(len(profiles), chunk_size)
    logging.info("Starting simulation with %d chunks", len(chunk_windows))

    def handle_interrupt(signum, frame):  # pragma: no cover - signal hook
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)

    with ProcessPoolExecutor(max_workers=len(chunk_windows)) as executor:
        try:
            while True:
                loop_start = time.perf_counter()
                futures = []
                for window in chunk_windows:
                    profile_chunk = [profiles[i] for i in window]
                    state_chunk = [states[i] for i in window]
                    futures.append(
                        executor.submit(
                            evolve_patient_batch,
                            profile_chunk,
                            state_chunk,
                        )
                    )

                total_payloads = 0
                for window, future in zip(chunk_windows, futures):
                    chunk_results = future.result()
                    for offset, (new_state, payload) in enumerate(chunk_results):
                        idx = window.start + offset
                        states[idx] = new_state
                        # Waveform snapshots are temporarily withheld from Pub/Sub.
                        pubsub_client.publish(
                            payload.to_dict(include_waveforms=False)
                        )
                        total_payloads += 1

                elapsed = time.perf_counter() - loop_start
                if elapsed > 60.0:
                    logging.warning(
                        "Simulation lagging: step took %.2fs (>60s)", elapsed
                    )
                else:
                    sleep_for = max(0.0, 60.0 - elapsed)
                    logging.info(
                        "Step emitted %d payloads in %.2fs; sleeping %.2fs",
                        total_payloads,
                        elapsed,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received; snapshotting state...")
            save_population_state(states)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def save_population_state(states: Sequence[PatientDynamicState]) -> None:
    """Persist current states for later resumption."""
    snapshot_path = SIMULATION_CONFIG.state_snapshot_path
    with snapshot_path.open("wb") as handle:
        pickle.dump(states, handle)
    logging.info("State saved to %s", snapshot_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    profiles, states = setup_population()

    pubsub_client = PubSubClient(PUBSUB_CONFIG)

    run_simulation_loop(profiles, states, pubsub_client)


if __name__ == "__main__":
    main()

