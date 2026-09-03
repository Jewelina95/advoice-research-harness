# ADvoice research harness

[![Reproducibility checks](https://github.com/Jewelina95/advoice-research-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Jewelina95/advoice-research-harness/actions/workflows/ci.yml)

ADvoice is an evidence-governed research pipeline for speech-based cognitive screening. It routes heterogeneous speech tasks, extracts acoustic, language, dialogue, and task measurements, converts them into typed `MetricEvidence`, aggregates them into cognitive `StateCards`, and constrains a single diagnostic Agent to cite the evidence it uses.

The repository compares three conditions:

| Condition | Description |
| --- | --- |
| `B1` | Traditional machine-learning baseline |
| `B2` | Direct LLM baseline |
| `B3 / Ours` | Supervised prior + cognitive evidence graph + constrained diagnostic Agent |

This software is for research on screening and referral support. It is not a diagnostic medical device and does not establish Alzheimer disease pathology or stage.

## Live repository demo

The public demo contains four deterministic, non-human synthetic audio fixtures:

1. Clinical interview
2. Picture description
3. Structured cognitive task
4. Natural speech

Each case demonstrates channel routing, feature extraction, `MetricEvidence`, `StateCards`, segment playback, trace mapping, and the evidence-constrained report contract. The static demo does not load trained clinical weights and does not call a live GPT model.

```bash
git clone https://github.com/Jewelina95/advoice-research-harness.git
cd advoice-research-harness
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make demo
```

Open `http://127.0.0.1:8765`.

The browser can also process a local WAV and transcript through the public evidence extractor. Uploaded files stay in the local Python process and are not sent to an external API.

## System pipeline

1. **Case routing** identifies dataset, task, language, modality, and speaker role.
2. **Input normalization** performs transcript normalization, participant isolation, ASR handling, and task-aware segmentation.
3. **Evidence construction** converts acoustic, language, dialogue, and task metrics into typed objects containing value, reference scope, direction, reliability, confounds, task, segment, and report permission.
4. **Cognitive-state construction** combines reliability-weighted evidence into shared and task-specific `StateCards` without allowing duplicate task views to cast duplicate votes.
5. **Supervised prior** learns text, audio, state, and segment branches with subject-level splits and out-of-fold predictions.
6. **Diagnostic Agent** reads a prior-blind evidence workspace and returns ordinal class evidence with valid source IDs. It does not write the final probability directly.
7. **Validation-frozen fusion** applies bounded correction only when coverage, reliability, confound, and routing gates pass; invalid outputs fall back to the supervised prior.
8. **Locked reporting** freezes the prediction before clinician-facing and participant-facing reports are rendered.
9. **Two-layer evaluation** measures prediction and calibration in Layer A, then evidence validity, traceability, report safety, ablations, and negative controls in Layer B.

## Model configuration

The model is not hidden inside the web page. Its configuration and implementation are versioned separately:

| Component | Location | What to change |
| --- | --- | --- |
| Prediction architecture | `configs/models/default.yaml` | encoders, folds, segment windows, fusion candidates |
| Agent provider and model | `configs/agents/default.yaml` | provider, model name, correction policy, retry rules |
| Evaluation endpoints | `configs/evaluation/default.yaml` | Layer A and Layer B metrics |
| Dataset adapters | `configs/datasets/*.yaml` | local raw path, tasks, labels, languages, exclusions |
| Training and constrained fusion | `src/advoice/condition_c.py` | Agent workspace, calibration, bounded correction |
| Statistical models | `src/advoice/models.py` | B1, B2, and supervised branches |
| Evidence and state construction | `src/advoice/evidence.py`, `src/advoice/states.py` | typed evidence and cognitive-state aggregation |
| Report renderer | `src/advoice/diagnostic_agent_report.py` | locked clinician and participant outputs |
| Public demo API | `demo/server.py` | local web endpoints and byte-range audio serving |

The default full-run Agent is configured in `configs/agents/default.yaml`. Changing the model or prompt contract creates a new experimental configuration and requires rerunning evaluation; it is not a cosmetic web change.

## Run one dataset

Raw data must be obtained from the dataset owner and mounted locally. Validate the adapter before training:

```bash
make validate DATASET=PREPARE_DrivenData
make full DATASET=PREPARE_DrivenData
make evaluate DATASET=PREPARE_DrivenData
make report DATASET=PREPARE_DrivenData
```

`NCMMSC2021_AD` uses long recordings only. Six-second clips and unlabeled test tracks are excluded by configuration.

The harness supports these independently evaluated dataset tasks:

- `IAEAV`
- `ADReSS_2020`
- `ADReSSo_2021_diagnosis`
- `ADReSSo_2021_progression`
- `PROCESS_2`
- `PREPARE_DrivenData`
- `TAUKADIAL`
- `DementiaBank_Pitt`
- `DementiaNet_PublicFigures`
- `NCMMSC2021_AD`

## PREPARE evaluation snapshot

The packaged web demo includes a frozen summary of the 9.2 PREPARE official-test run (`n=412`). ADvoice 9.2 reached accuracy `0.6723`, Micro AUROC `0.8462`, and Micro AUPRC `0.7011`. It exceeded both internal baselines, and the matched-encoder isolation estimated a Macro AUROC gain of `+0.0601` and accuracy gain of `+0.0752` from the cognition framework.

The standalone 9.2 result did **not** exceed the published SpeechCARE mean on its three headline endpoints. The release gate remains closed. The comparison is retrospective because the official test outcomes were inspected during development; it is not an untouched confirmatory validation.

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
demo/             four public fixtures, static outputs, local server, and web interface
schemas/          evidence, Agent, decision, trace, and report JSON contracts
src/advoice/      adapters, feature extraction, training, fusion, evaluation, and reports
tests/            contract, leakage, routing, correction, and demo tests
docs/             architecture, protocol, validation status, and reproducibility notes
```

Run the verification suite with:

```bash
make test
```

Current implementation status and known limitations are recorded in [docs/VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md).
