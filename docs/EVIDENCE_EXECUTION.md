# Evidence execution and independent Agent calibration

This change repairs the evidence-to-prediction contract. It does not establish a
clinical accuracy improvement, a newly trained GPT model, or superiority to
SpeechCARE. Licensed recordings and participant-level predictions remain local.

## Executed pipeline

1. Task/channel routing determines which configured measurements are applicable.
   An expected measurement absent from the feature table remains an explicit
   missing row. Unperformed tasks and disabled metrics are not invented.
2. Training-reference normalization and reliability produce metric evidence.
   Measurement absence and an unavailable reference distribution are separate
   fields. Neither is treated as a normal finding.
3. A shared aggregation function produces state cards. Model scores can include
   allowed predictive auxiliaries; report scores use only report-permitted
   measurements. Calibration and inference use the same aggregation. Original
   time spans are preserved separately from recomputed numeric scores.
4. Supervised module A learns the base fusion. Supervised module B learns a
   bounded residual correction from structured features. Prototype-distance
   features in this module are not GPT judgments or GPT post-training.
5. A blinded diagnostic Agent can propose evidence scores and state edits.
   Numerical Agent correction requires an independent development cohort,
   validated case-quality inputs, and the existing joint-gain acceptance gate.
6. A separate communication stage writes a report from the permitted, reviewed
   snapshot. Rejected evidence cannot reappear as clinical support.

## Independence boundary

For eligible datasets, the default configuration reserves 20% of the original
training participants before either supervised module selects or fits models.
The threshold is 120 training participants with sufficient class support. The
remaining fit partition supplies reference distributions and all supervised
selection. Original test participants remain untouched. Frozen encoder caches
can be reused only after their existing model/input fingerprint checks.

The model artifact and metadata record fit/calibration/test membership. Agent
calibration files explicitly declare the dedicated holdout. Runtime rejects
overlap, duplicate calibration participants, missing independence declarations,
and workspaces without matching provenance before requesting Agent calibration.
Selection-dependent historical OOF files are not accepted as independent data.

Small datasets still train the supervised pipeline, but do not gain permission
to change clinical probabilities using unvalidated Agent corrections. The size
threshold is an engineering policy, not a statistical power guarantee. Keeping
the original held-out split does not make repeatedly inspected test outcomes a
new confirmatory test.

## State edits and clinical publication

Validated edits create a separate reviewed snapshot. Invalidation removes the
affected state and linked evidence IDs; downweighting is explicit. Stale class
summaries are removed. The snapshot records its parent hash and actions.

**A state edit does not yet rerun the frozen supervised scorer.** Until an
executable, validated scorer replay is attached, the case is marked
`state_replay_required`, `prediction_released=false`, and
`released_probabilities=null`. The report states that the risk is withheld rather
than showing an unchanged probability as if it were recalculated. API-backed
report generation cannot bypass this restriction. These cases also cannot
calibrate numeric Agent corrections using the stale prior.

The original numerical prior is retained in the full-cohort evaluation table.
It is an evaluation fallback, not a released clinical result. This prevents
improving accuracy by silently deleting difficult cases. Report both all-case
performance and release coverage; any released-subset result needs its own
denominator and cannot replace the all-case result.

## How accuracy is measured

For single-label classification, accuracy is the number of correctly classified
held-out participants divided by all eligible held-out participants. Decisions
use the saved class order and maximum predicted probability. Multiclass
micro-F1 equals accuracy only under that same single-label, all-class convention.
Macro-F1 and class-wise sensitivity show minority-class errors that accuracy
can hide. AUROC evaluates ranking, not the chosen decision threshold. Calibration
and clinical report grounding are different outcomes and must not be conflated.

Comparisons require the same participants, label mapping, modalities, split and
aggregation. A processed-input pilot reuses preprocessing and B1/B2 artifacts;
it is not an end-to-end retraining or a live GPT experiment when the provider is
disabled. No held-out accuracy is used to select this repair.

## Remaining research work

- Attach frozen scorer replay to validated state edits and verify that a no-op
  replay exactly reproduces the saved prior before accepting an edited replay.
- Define and validate the case-level confound assessment producer. Current
  configuration warnings are not observed patient findings; do not turn unknown
  confounds into fabricated absence merely to activate a correction gate.
- Evaluate the full Agent path on fresh development data, with unchanged
  encoders for the mechanism comparison and an untouched external cohort for
  confirmation. These integrity tests alone do not establish clinical benefit.
- Preserve trajectory links as measurement provenance, not proof that every
  displayed segment causally changed the classifier.
