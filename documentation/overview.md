# Project Overview

The **T1D Adverse Data Generator** builds a high-fidelity digital twin of people
living with diabetes. Every minute, the simulator advances a cohort of virtual
patients, evolves their metabolic state, renders cardiovascular waveforms, and
streams sensor-grade payloads to Google Cloud Pub/Sub for downstream analytics.

## Primary Objectives

- Capture realistic physiological variability by combining stochastic
  demographics with circadian-aware behavior models.
- Emit data in a schema that maps directly onto warehousing targets such as
  BigQuery while remaining easy to replay locally.
- Provide a tunable sandbox for edge cases (e.g., hypoglycemia, stress events,
  sensor drift) that are expensive or impossible to collect from human studies.

## High-Level Workflow

1. `main_driver.py` bootstraps the cohort via `setup_population`, optionally
   restoring serialized dynamic states from disk.
2. The driver shards the cohort into `chunk_size` windows and hands each window
   to `simulation_engine.evolve_patient_batch` inside a process pool.
3. `process_single_patient` composes behavior signals from:
   - `simulation.activity_model` for circadian-driven movement intensity.
   - `simulation.humanization.scheduler` for stochastic meals and weekend drift.
   - `simulation.metabolic_model` for glucose/insulin kinetics.
   - `simulation.signal_model` for cardiovascular/autonomic targets and
     waveform synthesis via NeuroKit2.
4. Synthetic payloads are wrapped in the `data_models.SensorPayload` dataclass
   and either published to Pub/Sub or echoed to stdout, depending on
   configuration.

## Repository Map

- `config.py` – Centralizes simulation knobs (population size, chunk size,
  snapshot location) and Pub/Sub credentials sourced from the environment.
- `data_models.py` – Houses strongly typed dataclasses for demographics,
  per-minute state, and serialized sensor payloads, including helper factories
  such as `PatientStaticProfile.generate_random`.
- `simulation_engine.py` – Minute-wise integrator that ties together the
  various physiology sub-models and maintains simulation continuity.
- `simulation/` – Modularized sub-systems for activity inference, metabolic
  transitions, waveform generation, and humanization helpers (circadian drift,
  meal schedulers, sensor variability).
- `main_driver.py` – Production entry point that manages multiprocessing,
  Pub/Sub publishing, and graceful shutdown/snapshotting.

## Extending the Simulator

- **New biomarkers**: Extend `simulation.signal_model` to compute additional
  vitals and add new keys to `SensorPayload.vitals` or `SensorPayload.wearable`.
- **Alternative transports**: Implement another publisher class beside
  `PubSubClient` and call it inside `main_driver.py` based on configuration.
- **Cohort seeding**: Override `setup_population` to ingest real-world baseline
  data or to replay previously serialized `PatientStaticProfile` objects.

Refer to the other documents in this folder for environment setup,
simulation-flow specifics, and cloud pipeline details.

