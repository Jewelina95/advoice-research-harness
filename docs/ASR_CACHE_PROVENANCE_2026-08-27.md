# ASR cache provenance, 2026-08-27

## Purpose

The 8.27 full study reuses ASR outputs only when the underlying audio and the
ASR protocol are unchanged. This avoids recomputing thousands of deterministic
transcriptions while keeping all feature extraction, supervised training,
diagnostic-Agent execution, report generation, and evaluation in the 8.27 run.

## Cache identity contract

An ASR entry is reusable only when its SHA-256 cache key matches exactly. The
key contains:

- resolved audio path;
- audio file size and nanosecond modification time;
- patient analysis intervals;
- ASR backend;
- ASR model revision/name;
- configured language.

The current protocol is `mlx_whisper` with
`mlx-community/whisper-large-v3-turbo` and automatic language detection.

## Verified migration

Source cache:

`<local-project-root>/artifacts_legacy_8_13`

Exact matches against the newly built 8.27 manifests:

| Dataset | Current manifest rows | Exact matching ASR entries | Included in analysis |
|---|---:|---:|---:|
| ADReSSo 2021 progression | 47 | 47 | 47 |
| PREPARE DrivenData | 2,034 | 2,034 | 2,034 |
| TAUKADIAL | 507 | 507 | 507 |
| NCMMSC2021 AD long-speech track | 399 | 399 | 399 |

The legacy PREPARE cache contains 2,058 files because it also covers records
outside the current manifest. Only cache keys requested by the current 2,034-row
manifest are read by the pipeline.

## NCMMSC exclusion

The NCMMSC manifest uses only `AD_dataset_long/train` and the labelled long-form
test directory. `AD_dataset_6s` and the unlabelled test directory are excluded
by configuration. No six-second clip is an analysis input.

## What is not reused

No legacy label, split assignment, feature table, state score, model parameter,
prediction, Agent decision, clinical report, or evaluation metric is accepted
as a result of this cache migration. Those outputs are rebuilt by the 8.27
pipeline. Cache reuse therefore changes runtime, not the statistical protocol.
