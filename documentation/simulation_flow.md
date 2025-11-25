# Simulation Flow

This document details how a single 60-second tick moves through the stack,
highlighting the most relevant modules along the way.

## 1. Cohort Bootstrap (`main_driver.setup_population`)

- Seeds the RNG (`numpy.random.default_rng`) to guarantee reproducibility.
- Instantiates `PatientStaticProfile` objects via
  `PatientStaticProfile.generate_random`, ensuring internally consistent
  demographics (age → height → BMI → insulin sensitivity).
- Creates matching `PatientDynamicState` structures with initial timestamps,
  fasting metabolic buffers, and reset meal flags.

**Observability:** Summary logs report mean age and the percentage of insulin
resistant patients to keep cohort quality in check.

## 2. Workload Partitioning (`_chunk_indices` + `ProcessPoolExecutor`)

- The population index set is sliced into `chunk_size` windows stored in
  `chunk_windows`.
- Each iteration of the main loop uses a `ProcessPoolExecutor` with
  `len(chunk_windows)` workers so every chunk advances in true parallelism on
  multi-core hosts.
- Worker futures return ordered tuples of `(PatientDynamicState, SensorPayload)`
  per patient, preserving deterministic mapping back onto the shared `states`
  list.

## 3. Per-Patient Minute Evolution (`simulation_engine.process_single_patient`)

1. **Activity inference:** `simulation.activity_model.determine_activity`
   combines a bimodal circadian drive with individual noise to derive both an
   `ActivityMode` enum and a normalized intensity scalar.
2. **Humanization layers:**
   - `humanization.scheduler.adjust_for_weekend_wake_shift` delays wake drive on
     weekends.
   - `humanization.scheduler.maybe_trigger_meal_event` injects stochastic meals
     within realistic windows while preventing duplicate lunches.
   - `humanization.circadian.apply_circadian_drift` adds sinusoidal variation to
     vitals and thermoregulation.
   - `humanization.sensor_variability.apply_sensor_noise` increases measurement
     noise when the patient is active.
3. **Metabolic solver:** `simulation.metabolic_model.calculate_next_glucose`
   advances glucose, insulin on board, and gut carbohydrates with simplified but
   physiologically grounded heuristics (gastric emptying, basal insulin, hepatic
   glucose release).
4. **Vitals & wearables:**
   - `simulation.signal_model.calculate_vitals_target` maps glucose stress and
     activity into HR/HRV/QT/EDA targets.
   - `simulation.signal_model.generate_waveform_snapshot` renders 10-second ECG
     and EDA strips using NeuroKit2.
   - Wearable metrics such as steps per minute, skin temperature, and accelerometer
     magnitude are synthesized with added Gaussian noise via `simulation.noise`.
5. **State rollover:** `dataclasses.replace` is used to copy forward immutable
   fields (timestamp, tick counter, fatigue accumulation, metabolic buffers,
   meal flags) without mutating the incoming `PatientDynamicState`.
6. **Payload assembly:** All partitions are wrapped in `SensorPayload`, which
   exposes `to_dict(include_waveforms=bool)` for transport-specific serialization.

## 4. Publishing Loop (`run_simulation_loop`)

- After collecting chunk results, `states[idx]` is updated with the new
  `PatientDynamicState` to keep local memory coherent.
- `SensorPayload.to_dict(include_waveforms=False)` is used to minimize Pub/Sub
  message size. You can flip the flag to propagate waveform snippets when the
  topic supports larger payloads.
- Throughput logging highlights how many payloads are emitted per minute and
  whether the simulation step exceeded the real-time 60-second target. When the
  workload finishes early, the loop sleeps for the remaining slack to maintain
  wall-clock alignment.

## 5. Shutdown & Recovery

- SIGINT triggers a `KeyboardInterrupt`, which `run_simulation_loop` intercepts
  to call `save_population_state`.
- The snapshot is a pickle of `PatientDynamicState` objects stored at
  `SIM_STATE_SNAPSHOT_PATH`. You can load it manually in a modified
  `setup_population` to resume long-running experiments.

## Tuning Considerations

- **Population size vs. chunk size:** Increasing `SIM_POPULATION_SIZE` scales
  linearly. When hitting CPU limits, reduce `SIM_CHUNK_SIZE` to spawn more
  workers with smaller slices, improving cache locality.
- **Waveforms:** Generating ECG/EDA snippets dominates CPU time. Disable them or
  reduce `duration_seconds` in `signal_model.generate_waveform_snapshot` for
  lightweight runs.
- **Sensor fidelity:** Noise helpers (`simulation.noise`, `humanization.sensor_variability`)
  are centralized, making it easy to calibrate accuracy per signal without
  editing the rest of the pipeline.

