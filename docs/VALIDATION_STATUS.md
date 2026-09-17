# Validation status

Current engineering review: 2026-09-17. Historical results retain their original
protocol; the corrected implementation is under review on a separate Git branch.

## Current review

- 323 automated tests passed locally with plugin autoload disabled.
- Four public synthetic examples executed successfully, without clinical diagnosis.
- Four processed-input task pilots retrain condition C, with frozen historical
  preprocessing/B1/B2 and no live Agent: ADReSS2020 internal holdout, ADReSSo
  progression, NCMMSC long recordings, and PublicFigures.
- Eight fixed PREPARE validation candidates were trained; one locked candidate
  was tested once. Accuracy 0.665049 does not beat historical independent ADvoice
  (0.672330) or the SpeechCARE paper mean (0.7211). Do not promote this candidate.
- Corrections address fold references/reliability/temperature, evidence IDs,
  invalidated evidence, calibrated routing, batch case identity and metric input
  validation. These are implementation repairs, not evidence of clinical benefit.
- New real-time Agent prediction is not validated. Ordinary evidence workspaces
  lack an assessed case-level confound record, so predictive correction is closed.
- Eligible datasets now reserve independent Agent calibration participants before
  supervised model selection. Historical selection-dependent calibration files
  fail closed. Calibration and inference share state aggregation; expected missing
  evidence remains in coverage denominators, and coverage is batch-independent.
- Validated state edits create a reviewed snapshot and withhold clinical risk
  until scorer replay is implemented. API report writing cannot bypass this.
  State scorer replay and validated confound assessment remain release blockers.
- An independent reviewer reproduced two additional faults: relocated training
  inputs lost the audio-manifest path, and non-finite measurements became clipped
  abnormal state scores. Regression tests now cover both fixes.

Local runtime: Python 3.11, Torch 2.2 on macOS; not the declared Torch >=2.4
dependency floor. Pandas optional-dependency and SciPy deprecation warnings were
recorded. The old threadpoolctl/OpenBLAS probe needs the bounded environment-limit
fallback in the new PREPARE pilot. No global environment was upgraded. Clean
declared-dependency CI remains a separate requirement.

See [System review](SYSTEM_REVIEW.md) and [Research workspace](RESEARCH_WORKSPACE.md)
for the actual scope, remaining conditions, and reproducible commands.

## Completed under the 9.2 protocol

- PREPARE engineering run, method audit and SpeechCARE-aligned retrospective comparison.
- Same-encoder isolation of the cognitive representation contribution.
- Agent interface, evidence identifier, permission, fallback and calibration tests.
- Public synthetic demo and local upload API.
- 127 automated tests.

## Not completed under the 9.2 protocol

At the original 2026-09-03 freeze, nine other configured tasks had only historical
8.27 artifacts. Four now have the bounded processed-input pilots described above;
this does not replace all historical datasets with a fresh full-Agent evaluation.

The standalone PREPARE 9.2 system does not yet exceed the published SpeechCARE means on Micro AUROC, Micro F1 and Micro AUPRC. The release gate therefore remains closed. The repository records this status rather than changing thresholds or reusing held-out outcomes to manufacture a pass.

The PREPARE official test outcomes have been inspected repeatedly during development. A future model frozen after this point requires a new site, time or held-out cohort for confirmatory external validation.
