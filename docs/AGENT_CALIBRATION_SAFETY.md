# Agent calibration safety

## Two-stage selection

`fit_agent_two_stage_strengths` fails closed with both strengths zero when any
of HC, MCI, or AD is absent, including an empty development cohort. The audit
records class counts, missing classes, and `failed_closed_missing_classes`;
undefined baseline metrics are null, not fabricated values or reduced-class
AUROC. The held-out Agent gate cannot accept this selection status.

Nonzero two-stage corrections must satisfy all four development criteria:

- The existing minimum macro-F1 gain.
- The existing macro-AUROC noninferiority margin.
- Log loss no greater than baseline plus the prespecified log-loss margin.
- Multiclass Brier no greater than baseline plus the prespecified Brier margin.

Both new margins default to zero (no deterioration). Optional agent settings
`diagnostic_agent_log_loss_noninferiority_margin` and
`diagnostic_agent_brier_noninferiority_margin` must be specified before the
development comparison, not chosen after inspecting candidate results. No
configuration defaults or F1/AUROC thresholds were changed. Margins must be
finite and nonnegative and are recorded with the selection result.

Log loss is mean negative natural log probability of the true class, with
machine-epsilon clipping. Brier is the mean **sum** of squared errors across
HC/MCI/AD (range 0 to 2), not the class-averaged version. Candidate and baseline
scores use the same case set and class order. Routing uses calibrated OOF
evidence. This is a development acceptance gate, not proof of clinical calibration.

## Potential confounds are not observed findings

The current evidence producer copies `confounds` from metric configuration into
`confound_tags`. Neither the producer nor `case_evidence_package.schema.json`
defines an assessed case-level confound record with finding status, provenance,
affected scope, and a validated burden rule. Audio/text reliability and QC values
do not establish absence of language, task, ASR, device, or other confounds.

Workspace construction retains `confound_tags` for compatibility, explicitly
labels them `potential_only`, and exposes `potential_confound_tags` as context
warnings. It records `observed_confound_findings: null`,
`confound_assessment_status: unknown`, and `confound_burden: null`.
An empty potential-tag list also means unknown, not assessed absence.

The correction gate is zero with reason `unknown_case_level_confounds`.
The Agent workspace loader enforces the same behavior for historical workspaces;
an old positive gate or zero numeric burden is not a validated assessment.
Historical files themselves are unchanged. There is deliberately no new
configuration switch that converts unknown findings into absence.

The workspace API now accepts an optional typed `CaseConfoundAssessment`:
`{"status": "assessed", "burden": 0.2, "provenance": "assessment artifact ID"}`.
This is an interface example, not an observed assessment. Burden must be a finite
number in [0, 1] (not a boolean), with nonempty provenance and assessed status.
Historical workspaces cannot enable the gate using an old numeric value alone.

The current production evidence producer does not generate validated assessments.
Consequently predictive Agent corrections remain closed on its ordinary outputs.
Evidence remains available for review, and the supervised prior is preserved.
Re-enabling correction requires a validated case-level assessment producer and a
versioned burden rule, followed by fresh development calibration. The typed
interface validates structure, not medical truth. Merely supplying a string or
renaming potential tags is insufficient.

This patch does not change the separate static-tag feature in Condition C's
supervised quality frame; that remains with its training owner. No pipeline,
configuration, training, or cached result files were modified by this change.

## State execution boundary

State updates still enforce evidence permissions rather than rebuilding states
or rerunning supervised heads. Real state reaggregation remains a separate
scorer design. The calibration and confound changes do not claim to implement it.
