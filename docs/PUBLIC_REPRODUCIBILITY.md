# Public reproducibility workflow

## What can be reproduced without clinical data

The synthetic demo runs the same low-level audio and transcript feature extractor used by the research harness, constructs typed MetricEvidence records, aggregates three illustrative cognitive StateCards and renders the evidence trace. It does not load a trained diagnostic model and does not emit disease probability or stage.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
make demo
```

Open `http://127.0.0.1:8765`. The page can run the packaged sample or accept a local WAV and transcript. Uploaded material is processed by the local Python process and is not sent to an external API.

## Local four-channel case gallery

Researchers with authorized local access can generate an ignored manifest that points to one pseudonymized AD-labelled example from each established data channel:

```bash
python demo/build_local_case_manifest.py --advoice-root "/path/to/AD voice"
make demo
```

The four channels are clinical interview (IAEAV), standard picture description (ADReSS 2020), structured cognitive tasks (PROCESS-2), and non-standard public speech (DementiaNet). Channel routing changes the visible evidence set: interviews add dialogue burden measures; picture tasks add content-unit and information-density measures; public speech adds auxiliary prosody and recording-quality measures that are not marked as reportable disease evidence. The server analyzes these files on demand with the current feature code and supports HTTP byte ranges for audio playback and seeking. This gallery demonstrates channel routing and evidence traceability; it is not a replacement for dataset-level 9.2 evaluation.

`demo/local_cases.json`, `demo/local_output/`, source identifiers, transcripts, and audio remain local-only and are never committed.

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
