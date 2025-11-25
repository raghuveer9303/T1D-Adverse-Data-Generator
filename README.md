# T1D-Adverse-Data-Generator

Synthetic digital twin that simulates cohorts of people living with diabetes,
streaming minute-level vitals, metabolic markers, and wearable data to Google
Cloud Pub/Sub or stdout for downstream experimentation.

## Documentation

- `documentation/overview.md` – Conceptual overview, goals, and repository map.
- `documentation/setup_and_usage.md` – Environment preparation and runtime
  instructions.
- `documentation/simulation_flow.md` – Detailed walkthrough of the per-minute
  evolution loop and supporting sub-models.
- `documentation/cloud_pipeline.md` – Pub/Sub integration details and payload
  schema guidance.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy neurokit2 google-cloud-pubsub python-dotenv
cp env.sample .env  # update values as needed
python main_driver.py
```

Set `ENABLE_PUBSUB=false` in your environment to run locally and echo payloads
instead of publishing them.
