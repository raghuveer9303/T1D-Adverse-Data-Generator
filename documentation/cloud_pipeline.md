# Cloud Pipeline

The simulator natively streams sensor payloads to Google Cloud Pub/Sub, allowing
real-time dashboards, anomaly detectors, and warehousing jobs to ingest rich
digital twin data. This document explains how transport decisions are made and
how to consume the emitted schema.

## Pub/Sub Client Lifecycle

- `main_driver.PubSubClient` is a thin wrapper around
  `google.cloud.pubsub_v1.PublisherClient`.
- When `ENABLE_PUBSUB=false` *or* the dependency is missing, the publisher is
  left as `None`. In that case `publish()` simply prints JSON payloads to
  stdout, which keeps local development frictionless.
- With Pub/Sub enabled, the client batches up to 100 messages or 1 second of
  latency per API call (via `BatchSettings`) to keep network overhead low even
  when simulating large populations.

## Message Schema (Derived from `SensorPayload`)

Each Pub/Sub data attribute is JSON-encoded UTF-8 bytes. A representative
payload looks like:

```json
{
  "meta": {
    "message_id": "c82d1f4e-7874-4d6c-8e41-8388876c1b7f",
    "timestamp": "2025-11-25T16:40:00+00:00",
    "device_id": "cgm-2f40c5fd"
  },
  "vitals": {
    "heart_rate_bpm": 92,
    "hrv_sdnn": 28.5,
    "qt_interval_ms": 410,
    "spo2_pct": 96.4,
    "resp_rate_rpm": 22
  },
  "metabolics": {
    "glucose_mgdl": 154,
    "trend_arrow": "RISING"
  },
  "wearable": {
    "steps_per_minute": 84,
    "accel_y_g": 0.12,
    "skin_temp_c": 33.7,
    "eda_microsiemens": 6.1
  }
}
```

> Note: `waveform_snapshots` is omitted from Pub/Sub by default to keep message
> size manageable. Set `include_waveforms=True` when calling `to_dict` if your
> analytics pipeline requires raw ECG/EDA samples.

### Field Conventions

- Timestamps are ISO 8601 strings with timezone offsets.
- Enum values (e.g., `ActivityMode`) are serialized to their `.value` strings.
- Numeric noise is intentionally added to mimic imperfect sensors; downstream
  filters should expect occasional missing values when sensor noise helpers
  simulate dropouts.

## Consuming the Stream

- **BigQuery:** Use `STRUCT` columns mirroring the JSON schema or flatten the
  payload with `bq load ... --source_format=NEWLINE_DELIMITED_JSON`.
- **Dataflow / Beam:** Decode each Pub/Sub message into a Python dict and route
  on `meta.device_id` or `metabolics.trend_arrow` for cohort slicing.
- **Real-time dashboards:** Because payloads arrive once per minute per patient,
  simple rolling windows (5–15 minutes) are sufficient to reconstruct vitals
  timelines.

## Reliability & Failure Modes

- If Pub/Sub publish futures fail, the current implementation simply drops the
  exception (callback is a no-op). Extend `PubSubClient.publish` to attach error
  handlers or retries if your deployment requires stronger guarantees.
- When the publisher is unavailable, stdout logging acts as a poor-man's dead
  letter queue. You can pipe the output into `jq`, write it to disk, or forward
  it into another message bus.
- Snapshotting (`save_population_state`) is independent of Pub/Sub; even if the
  transport layer is down, you can persist the cohort locally and replay it
  later once infrastructure issues are resolved.

## Extending the Pipeline

- **Multiple topics:** Instantiate additional `PubSubClient`s with different
  topic IDs to broadcast subsets of the payload (e.g., high-frequency vitals vs.
  aggregated trends).
- **Edge deployments:** Replace Pub/Sub with MQTT, Kafka, or WebSockets by
  implementing a drop-in replacement exposing a `publish(dict) -> None` method.
- **Data governance:** Attach Cloud Logging, Data Loss Prevention (DLP), or
  Pub/Sub Lite depending on compliance and throughput requirements.

