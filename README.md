# ADvoice Research Harness

This repository replaces date-stamped script copies with a traceable, configuration-driven pipeline. A change to data, metric definitions, state definitions, fusion, agent prompts, evaluation, or report templates invalidates only the affected downstream stages.

The repository stores code and definitions. Raw audio, ASR text, model artifacts, case reports and immutable run records stay local and are excluded from Git.

## Fixed experimental contract

- **B1**: traditional acoustic machine learning. It uses hand-crafted acoustic features and does not generate a clinical report.
- **B2**: real direct transcript agent. It receives only de-identified ASR text and a fixed prompt. A failed or unavailable agent is reported as `not_run`; no deterministic proxy is substituted.
- **Ours**: MetricEvidence -> StateCard -> reliability-conditioned hierarchical fusion -> calibrated probability. A report agent translates the frozen structured evidence; it cannot change the numeric probability. Whether the gate actually becomes case-adaptive is audited from the learned coefficient and test-set weight variance.

## One-command usage

```bash
make validate
make quick
make full
make report
make test
```

`RUN_FULL.command` runs the full NCMMSC2021_AD pipeline and opens the latest report. Stable output locations are:

- `reports/latest/system_report.html`
- `reports/latest/evaluation_report.html`
- `reports/latest/index.html`

Immutable provenance is stored under `runs/<run_id>/run_manifest.json`. Raw clinical audio remains local and is linked under `data/raw`; it is never committed to Git.

`quick` stops before ASR and agent stages. `full` runs B1, B2, Ours, negative controls, ablations, evaluation and both reports. Content hashes let unchanged upstream stages use cache. `report` rebuilds HTML from the latest immutable run without retraining.

## Current NCMMSC contract

- Uses only `AD_dataset_long`: 280 official training recordings and 119 labelled test recordings.
- Explicitly excludes all 3,621 six-second clips.
- Aggregates recordings by composite subject key before modeling.
- Audits subject overlap, label conflict, duplicate hashes and 16 kHz mono consistency.
- Disables unvalidated language, semantic, syntax and interaction states for Ours because this local release has no human transcript, answer-unit rubric or speaker-role annotation.
- B2 receives generated Whisper ASR only; its input and output are cached separately from the report agent used by Ours.

## Project layout

```text
configs/             dataset, metric, state, model, agent and evaluation contracts
data/raw/            ignored local links to clinical data
references/          reviewed 7.16 metric/state source tables
src/advoice/         executable pipeline stages
templates/           stable system and evaluation HTML templates
tests/               unit tests for splits, evidence, states and evaluation
artifacts/            ignored mutable stage cache
runs/<run_id>/        ignored immutable run snapshot and manifest
reports/latest/       ignored stable user-facing HTML output
```

## Updating the system

- New dataset: add `configs/datasets/<dataset>.yaml` and a manifest adapter in `src/advoice/data.py`.
- Metric change: edit `configs/metrics/audio_metrics.yaml`.
- State change: edit `configs/states/audio_states.yaml`.
- Fusion change: edit `src/advoice/models.py` or `configs/models/default.yaml`.
- Agent change: edit `configs/agents/default.yaml`.
- Evaluation change: edit `configs/evaluation/default.yaml`.

The run manifest records configuration hashes, source hashes, dataset inventory, package versions, stage cache decisions, model parameters, and generated artifacts.

## Scientific interpretation boundary

The current NCMMSC run does not establish that Ours has better discrimination than B1. Ours improves calibration and structured traceability, while B1 currently has higher test AUROC and accuracy. The QC-only negative control is also strong, so cross-device or clinical robustness cannot be claimed without source-stratified and external validation. These findings are rendered directly in the evaluation HTML instead of being overwritten by a preferred narrative.

The automated clinical-report score is a structural audit, not a completed clinician study. Human evaluation requires a blinded rubric, multiple clinicians and inter-rater agreement.
