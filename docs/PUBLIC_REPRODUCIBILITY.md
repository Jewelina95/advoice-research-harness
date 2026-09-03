# Public reproducibility workflow

## What can be reproduced without clinical data

The synthetic demo runs the same low-level audio and transcript feature extractor used by the research harness, constructs typed MetricEvidence records, aggregates three illustrative cognitive StateCards and renders the evidence trace. It does not load a trained diagnostic model and does not emit disease probability or stage.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
make demo
```

Open `http://127.0.0.1:8765`. The page can run the packaged sample or accept a local WAV and transcript. Uploaded material is processed by the local Python process and is not sent to an external API.

## Full research run

1. Obtain the datasets under their original licenses.
2. Mount them under `data/raw/` or change the relative `raw_path` in the relevant dataset YAML.
3. Validate one dataset with `make validate DATASET=ADReSS_2020`.
4. Run one full dataset with `make dataset DATASET=ADReSS_2020`.
5. Run `make evaluate DATASET=ADReSS_2020` and `make report DATASET=ADReSS_2020`.

The PREPARE release gate is a retrospective engineering gate because its official test outcomes have already been inspected during development. It cannot be described as untouched confirmatory external validation.

## API

The local demo server exposes:

- `GET /api/sample`: packaged synthetic result.
- `GET /api/sample-audio`: packaged synthetic WAV.
- `POST /api/analyze`: JSON containing `audio_base64`, `transcript`, `task_type` and `language`.

The upload limit is 20 MB. The endpoint is a local demonstrator, not a hardened clinical service.
