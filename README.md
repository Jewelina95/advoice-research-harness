# ADvoice research harness

[![Reproducibility checks](https://github.com/Jewelina95/advoice-research-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Jewelina95/advoice-research-harness/actions/workflows/ci.yml)

ADvoice is an evidence-governed research pipeline for speech-based cognitive screening. It routes heterogeneous speech tasks, extracts acoustic, language, dialogue, and task measurements, converts them into typed `MetricEvidence`, aggregates those objects into cognitive `StateCards`, and constrains one diagnostic Agent to use traceable evidence.

This software is for screening and referral-support research. It is not a diagnostic medical device and does not establish Alzheimer disease pathology or stage.

## Run the demonstration

The web demonstration has two views:

- **Single recording:** run one packaged recording through routing, evidence construction, state formation, and the clinician-facing report contract. Evidence and state views remain linked to the source segments.
- **Cohort result:** inspect the held-out ADReSS 2020 result from one archived ADvoice run, including discrimination, class-level behavior, calibration, confusion matrix, and evidence-integrity checks.

```bash
git clone https://github.com/Jewelina95/advoice-research-harness.git
cd advoice-research-harness
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make demo
```

Open `http://127.0.0.1:8765`.

The four packaged recordings are deterministic synthetic fixtures rather than participant recordings. The browser can also process a local WAV and transcript through the public evidence extractor. Local uploads stay in the Python process and are not sent to an external service.

## What runs

1. **Route the case** by dataset, task, language, modality, and speaker role.
2. **Normalize the input** with participant isolation, transcript normalization, and task-aware segmentation.
3. **Construct evidence** by converting acoustic, language, dialogue, and task measurements into typed objects with values, reference scopes, directions, reliability, confounds, task IDs, segment IDs, and report permissions.
4. **Form cognitive states** by combining non-duplicated evidence into shared and task-specific `StateCards`.
5. **Estimate class evidence** with supervised text, audio, state, and segment branches trained under subject-level splits.
6. **Run constrained review** with a prior-blind evidence workspace. The diagnostic Agent returns class evidence and source IDs, not an unrestricted diagnosis.
7. **Apply frozen fusion rules** so correction occurs only when coverage, reliability, confound, and routing gates pass.
8. **Render the report** after the prediction is locked, preserving links from findings to states, metrics, and source segments.

When served by `demo/server.py`, pressing **Run selected recording** executes routing, feature extraction, evidence construction, and state formation again. Static hosting falls back to the frozen deterministic result. The packaged synthetic cases demonstrate the interface contract without trained clinical weights or a live GPT call; full dataset experiments use the versioned model and Agent configurations below.

## Code and configuration

| Component | Location | Responsibility |
| --- | --- | --- |
| Pipeline entry point | `src/advoice/pipeline.py` | dataset run orchestration |
| Evidence objects | `src/advoice/evidence.py` | measurement-to-evidence conversion |
| Cognitive states | `src/advoice/states.py` | shared and task-specific state aggregation |
| Training and fusion | `src/advoice/condition_c.py` | out-of-fold training, constrained Agent fusion, fallback |
| Diagnostic Agent | `src/advoice/cognitive_agent.py` | evidence workspace and structured Agent response |
| Prediction architecture | `configs/models/default.yaml` | encoders, folds, windows, and fusion candidates |
| Agent configuration | `configs/agents/default.yaml` | provider, model, correction policy, and retries |
| Dataset adapters | `configs/datasets/*.yaml` | local paths, tasks, labels, languages, and exclusions |
| Evaluation contract | `configs/evaluation/default.yaml` | predictive and evidence-integrity endpoints |
| Clinician report | `src/advoice/diagnostic_agent_report.py` | locked report rendering |
| Demo API | `demo/server.py` | local API and byte-range audio serving |

Changing a model, prompt contract, feature definition, or correction rule creates a new experimental configuration and requires a new evaluation run.

## Run a cohort

Raw data must be obtained from the dataset owner and mounted locally. The public cohort demonstration uses ADReSS 2020:

```bash
make validate DATASET=ADReSS_2020
make full DATASET=ADReSS_2020
make evaluate DATASET=ADReSS_2020
make report DATASET=ADReSS_2020
```

The archived held-out result shown in the web interface contains 27 participants: accuracy `0.8519`, balanced accuracy `0.8489`, macro F1 `0.8500`, macro AUROC `0.9780`, and ECE `0.1186`. It is the strongest archived cohort result currently available for this public interface. It is a development artifact, not a new confirmatory clinical validation, and should be replaced by a locked rerun before manuscript reporting.

The harness also has adapters for IAEAV, ADReSSo 2021 diagnosis and progression, PROCESS-2, TAUKADIAL, DementiaBank Pitt, DementiaNet public figures, and NCMMSC2021-AD. Restricted datasets are evaluated independently; participant-level raw data are never bundled in this repository.

## Data boundary

No participant recording, protected transcript, clinical label, or identifiable metadata is committed. The four public WAV files are deterministic tones and silence generated by `demo/generate_sample.py`; they are not human speech and their reference values are not clinical norms.

Authorized researchers can create an ignored local manifest that points to restricted examples without copying them into the repository:

```bash
python demo/build_local_case_manifest.py --advoice-root "/path/to/AD voice"
make demo
```

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for dataset-specific boundaries.

## Repository map

```text
configs/          versioned project, dataset, model, Agent, and evaluation settings
demo/             packaged fixtures, static outputs, local server, and web interface
schemas/          evidence, Agent, decision, trace, and report JSON contracts
src/advoice/      adapters, features, training, fusion, evaluation, and reports
tests/            contract, leakage, routing, correction, and demo tests
docs/             architecture, protocol, validation status, and reproducibility notes
```

Run the verification suite with:

```bash
make test
```

Current implementation status and known limitations are recorded in [docs/VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md).
