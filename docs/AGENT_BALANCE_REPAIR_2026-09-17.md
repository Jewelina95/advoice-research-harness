# Agent and supervised evidence: repair and validation plan

Status: engineering candidate; no new training or live patient inference in this change.

## What balance means

Keep the trained predictor as a statistical reference. The Agent receives the
transcript, task-conditioned MetricEvidence, StateCards, source references and
AD skills, and records a blind assessment before seeing predictor outputs.
Its evidence revisions are compiled and replayed through the state predictor.
The final decision combines the reference with admissible new information.
An Agent score margin is an evidence preference, not a calibrated probability
or an independently verified measure of clinical reliability.

A stronger general model does not justify increasing a constant Agent weight.
Authority must be earned on subject-disjoint development cases for the exact
model, skill, language, task and endpoint. The current coefficients remain
explicit pilot settings; this repair does not claim they are learned or optimal.

## Current decision sequence

1. Route the observation task and endpoint; preserve roles and task boundaries.
2. Compile observable metrics, reliability components and reference provenance.
3. Build StateCards and the traceable evidence workspace.
4. Obtain the Agent's blind structured assessment and ordinal class scores.
5. Show the learned advisor outputs and obtain cited state revisions.
6. Validate and replay accepted revisions; calculate pre/post state change.
7. Fuse the original predictor, eligible state change and eligible blind scores.
8. Save the decision and audit. Generate a clinician narrative only when requested.

Steps 4 and 5 are two decision calls. Step 5 is not merely report generation;
removing it would remove the current state-revision mechanism. Narrative reports
must not be described as the reason for all second-pass latency.

## Implemented formula repair

The previous gate compared the absolute post-revision state class with the
original predictor, although the actual update used a pre/post likelihood ratio.
These can point in opposite directions. For example, impairment probability
falling from 0.95 to 0.70 still predicts impairment, but the revision supplies
evidence against impairment. The gate now examines the delta actually added.

Screening gates now use binary HC-versus-impairment uncertainty. A distribution
of HC=0.05, MCI=0.475, AD=0.475 is uncertain about stage but confident about
impairment. Three-class entropy previously unlocked weak HC counterevidence in
this situation. A new regression test reproduces and prevents that mistake.

When a non-neutral Agent assessment and a state delta have opposite preferences,
the state path is blocked and the conflict is recorded. The blind ordinal path
can act only under its existing evidence-margin and eligibility checks. This is
a conservative conflict policy, not proof that the Agent is correct. If the
Agent agrees with the reference, both correction paths can remain neutral.

When a state correction is admitted, the ordinal path remains off to avoid adding
the same Agent review twice. State evidence is bounded. Weak ordinal evidence
cannot receive a gate larger than its own normalized score margin. Equal
evidence or zero strengths preserve the original probabilities exactly.

The three-class implementation currently acts on HC versus MCI-or-AD and
preserves the MCI:AD odds. It cannot fix MCI/AD staging errors. Public speech
currently preserves predictor probabilities. These are explicit current
limitations, not newly demonstrated benefits or universal clinical rules.

## Verification and cache policy

Six new regression cases failed on the old implementation and pass after the
repair: absolute-class/delta mismatch, a useful delta before the state class
crosses its boundary, strong and weak opposite Agent assessments under an
uncertain predictor, binary state/Agent contradiction, and confusing stage
uncertainty with screening uncertainty. Formula version v3
must enter study identity so old completed outputs cannot be silently reused.

`scripts/replay_authority_fusion_audit.py` reads previously saved numeric inputs,
preserves their coefficients, and writes a separate diagnostic replay. It has no
provider client, no fitting step and consumes no outcome labels. Existing
outputs cannot be overwritten. Replayed cases are development diagnostics after
inspection and cannot be promoted to untouched test evidence.

## Window/Agent assignments

| Workstream | Model | Scope | Completion condition |
| --- | --- | --- | --- |
| Fusion implementation | Astra/main | Delta gate, conflict arbitration, numeric invariants | Reproduced failures pass; no hidden label dependence |
| Independent method review | Independent reviewer | Challenge authority, double counting and held-out validity | Specific findings resolved or explicitly retained as limitations |
| Decision/report boundary | Astra/Parfit | Active runner, report opt-in, formula cache identity, saved report example | Decision path has no narrative generation; report cannot alter decision |
| Validation calibration, next | Frozen chosen model | Fit small route-aware calibration using development/OOF outputs | Gain assessed with class recall, ranking and calibration, not accuracy alone |

## Next experiment, without restarting full training

1. Complete cached-input regression and independent review; freeze the formula.
2. Freeze a subject-disjoint development/calibration manifest. Previously viewed
   pilot cases are development cases. If an official test has informed changes,
   disclose that and obtain a fresh independent confirmation cohort.
3. On the same development cases, compare the supervised reference, blind Agent,
   replay-only model and the combined candidate. Keep encoders and manifests fixed.
4. Fit low-capacity nonnegative correction strengths and temperature, with
   shrinkage across task/language groups. Include zero correction as a candidate.
   Fit on out-of-fold development outputs; never select weights on test accuracy.
5. Give MCI/AD staging a separate calibrated component only if development cases
   contain enough stage-specific evidence. Do not silently reuse the screening
   gate or assume speech alone provides definitive clinical staging.
6. Predeclare log loss/Brier, AUROC, accuracy and class-recall acceptance margins.
   If accuracy rises while ranking/calibration deteriorate, retain the full result
   and investigate; do not select whichever metric improved after seeing test data.
7. Freeze model/skill/calibration artifacts before a complete PREPARE comparison.
   Sol versus Astra is a controlled validation factor, not a choice made per test case.

No conclusion that this patch surpasses SpeechCARE follows from unit tests or
the small inspected pilots. Its immediate contribution is correcting the
decision semantics and making further validation cheaper and reproducible.

## Verified outcome

Full local suite after integration: 766 passed, 3 skipped. The environment emits
existing numerical-library compatibility/deprecation warnings; these were not
silenced or treated as model performance evidence.

Label-free cached replay found four state/Agent conflicts among nine inspected
PREPARE cases and two among three inspected NCMMSC long-track cases. One saved
class changes in each replay relative to the previous fusion. This is a behavior
audit, not a new accuracy or superiority claim. It made zero provider calls and
did not retrain a dataset model.

Independent review accepted the delta/decision-space corrections and retained
two methodological blockers: the ordinal fallback lacks a fitted, version-bound
calibrator and explicit evidence eligibility beyond score margins; MCI/AD stage
odds remain frozen. Existing threshold values were preserved, not optimized on
the inspected cases. The repair is suitable for further development validation,
not automatic promotion to the clinical or publication benchmark path.
