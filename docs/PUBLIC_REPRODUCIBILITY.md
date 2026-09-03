# Public reproducibility workflow

## Packaged demonstration

The repository includes four deterministic synthetic fixtures covering clinical interview, picture description, structured cognitive task, and natural speech. Each fixture runs through the low-level audio and transcript extractor, channel-specific metric routing, typed `MetricEvidence`, cognitive `StateCards`, segment links, trace rendering, and the report output contract.

The public fixtures contain no participant data. They do not load trained diagnostic weights, call a live Agent, or produce disease probability.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make demo
```

Open `http://127.0.0.1:8765`.

The same page works as a static site using the committed JSON outputs. Static hosting cannot process uploaded files; the local server can process a WAV and transcript through `POST /api/analyze`.

## Full research run

1. Obtain each dataset under its original license.
2. Mount it under `data/raw/` or change the corresponding `raw_path` in `configs/datasets/`.
3. Run `make validate DATASET=<dataset>`.
4. Run `make full DATASET=<dataset>`.
5. Run `make evaluate DATASET=<dataset>`.
6. Run `make report DATASET=<dataset>`.

The full command loads the configured encoders, produces subject-level out-of-fold predictions, builds the Agent evidence workspace, invokes the configured Agent provider, applies validation-frozen constrained fusion, locks decisions, and renders reports. The public demo intentionally stops before these clinical prediction stages.

## Local restricted examples

Authorized researchers can generate an ignored manifest that points to selected local examples:

```bash
python demo/build_local_case_manifest.py --advoice-root "/path/to/AD voice"
make demo
```

`demo/local_cases.json`, `demo/local_output/`, source identifiers, transcripts, and audio remain local-only. They are never copied into the public repository.

## Local API

- `GET /api/cases`: packaged public cases plus any authorized local manifest entries.
- `GET /api/case/<case_id>`: one public or local evidence result.
- `GET /api/case-audio/<case_id>`: byte-range audio stream for playback and seeking.
- `GET /api/evaluation`: frozen PREPARE evaluation summary.
- `POST /api/analyze`: local WAV analysis using `audio_base64`, `transcript`, `task_type`, and `language`.

The upload limit is 20 MB. The server is a local research demonstrator, not a hardened clinical service.
