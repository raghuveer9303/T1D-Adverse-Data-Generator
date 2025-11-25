# Setup & Usage

This guide walks through environment preparation, credential management, and
day-to-day simulator operations.

## Prerequisites

- Python 3.10 or later (f-strings, `typing` features, and multiprocessing gains).
- `pip` and a virtual environment manager (`python -m venv` or `conda`).
- Google Cloud credentials with Pub/Sub Publisher permissions if you plan to
  stream data beyond the local machine.

## Install Dependencies

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the required packages:
   ```bash
   pip install numpy neurokit2 google-cloud-pubsub python-dotenv
   ```
   Add any project-specific utilities (e.g., `black`, `pytest`) as needed.

## Configure Environment Variables

The simulator loads configuration from the process environment. Copy the sample
file to `.env` (or export variables directly) and update the values:

```bash
cp env.sample .env
```

Key fields:

- `SIM_POPULATION_SIZE`: Number of concurrent digital twins to simulate.
- `SIM_CHUNK_SIZE`: Patients per worker process; smaller sizes improve load
  balancing at the cost of more processes.
- `SIM_STATE_SNAPSHOT_PATH`: Path used by `save_population_state`.
- `ENABLE_PUBSUB`: Set to `false` to run offline and print payloads to stdout.
- `GCP_PROJECT_ID` / `PUBSUB_TOPIC_ID`: Required when Pub/Sub is enabled.

When `.env` is present, `config.py` automatically loads it through `python-dotenv`.

## Running the Simulation

```bash
python main_driver.py
```

What happens:

1. Population bootstrap via `setup_population`.
2. `PubSubClient` initialization (connects to Pub/Sub if enabled/installed).
3. Infinite minute-by-minute loop that spawns `ProcessPoolExecutor` workers,
   aggregates payloads, and either publishes them or prints JSON locally.

### Graceful Shutdown & Snapshots

- Send `Ctrl+C` (SIGINT) to trigger `save_population_state`, which writes the
  current `PatientDynamicState` list to the configured snapshot path.
- Restarting currently always seeds a fresh population, but you can adjust
  `setup_population` to reload the snapshot if desired.

## Pub/Sub Credential Notes

- When running outside GCP, set `GOOGLE_APPLICATION_CREDENTIALS` to point at
  your service-account JSON (one sample is checked in for reference).
- Topic provisioning is external to this repo; create it with `gcloud pubsub
  topics create <topic>` and grant the service account the Publisher role.
- If the Pub/Sub dependency is missing, `PubSubClient` gracefully falls back to
  printing payloads instead of crashing.

## Local Debugging Tips

- Reduce `SIM_POPULATION_SIZE` to a single patient to inspect the raw payload
  stream.
- Toggle `include_waveforms=False` inside `run_simulation_loop` if you need to
  shrink message size while testing new schema fields.
- Use `population_state.pkl` to capture tricky trajectories and replay them
  deterministically while adjusting sub-model parameters.

