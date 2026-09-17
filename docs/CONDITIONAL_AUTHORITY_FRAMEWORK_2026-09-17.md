# ADvoice conditional-authority framework review

Date: 2026-09-17

## Executive decision

The current six-stage direction is valid, but the implementation is not yet a
closed diagnostic loop. The evidence representation is substantially stronger
than a direct multimodal classifier, while the authority boundary between the
supervised predictors and the Agent remains incomplete.

The next architecture should not assign one global percentage to the Agent and
one percentage to supervised learning. Authority must be conditional on the
case, task, language, evidence quality, distribution shift and disagreement
pattern. The primary effect of the Agent should be to review and revise the
evidence graph, followed by numerical replay through the frozen predictor. A
separate bounded residual is allowed only for information that was not already
consumed by the supervised model. This avoids counting the same MetricEvidence
twice.

The target system remains one diagnostic Agent, not a multi-Agent discussion.
It uses a versioned AD skill, structured evidence tools and a deterministic
validator. It does not fine-tune GPT on the patient data in this release.

## What the current system already does correctly

1. Dataset adapters retain subject, task, language, recording and available
   speaker-role information.
2. Channel profiles prevent states that cannot be measured by a task from being
   silently invented.
3. MetricEvidence records the observed value, training reference, abnormal
   direction, reliability, missingness, confounds, evidence role and report
   permission.
4. StateCards retain supporting metrics, counterevidence and source segments,
   and separate model-facing from clinician-facing evidence.
5. Task-specific metric and state instances are generated for multi-task data.
6. Training references are fitted from training controls and are rebuilt inside
   folds in the repaired training path.
7. The Agent receives a label-blinded workspace, must inspect quality and
   counterevidence, records a blind evidence hypothesis, and can inspect the
   transcript before reviewing frozen model outputs.
8. Benchmark and clinical modes are separate. The benchmark requires a class;
   the clinical mode may withhold a definitive class and recommend repeat
   assessment.

## Main gaps found in the current six stages

### 1. Channel and target are not fully separated

Picture description, interview, structured multi-task and spontaneous speech
are observation channels. Cross-sectional HC/MCI/AD classification and
longitudinal decline are prediction targets. `ADReSSo_2021_progression` is
currently represented as a channel, although its defining difference is the
longitudinal endpoint. A single visit cannot support a progression conclusion.

The revised router must therefore have two independent outputs:

- `observation_route`: what the recording can measure;
- `target_route`: what the model is allowed to predict.

### 2. Task observability is configured at channel level, not at task-instance level

The current channel profiles enable one state set for an entire dataset. This is
too coarse for PREPARE, PROCESS-2 and DementiaBank Pitt. Picture description,
verbal fluency, recall and reading do not observe the same states. The system
retains task-specific columns, but it still needs a task-by-state observability
matrix and task-specific validation rules.

### 3. Metric reliability is compressed too early

The current evidence table contains a scalar reliability, but the downstream
Agent cannot distinguish whether reliability was reduced by ASR quality, role
coverage, alignment, short duration, reference support or measurement
instability. The components must remain separate and versioned.

MetricEvidence also needs direct source segment IDs, reference sample size,
reference artifact identity, measurement version and an explicit distinction
between potential and observed confounds.

### 4. State formation does not yet implement correlation families

Static weights reduce duplication, but they do not prove that several pause
metrics are independent evidence. Highly correlated metrics should first be
grouped into measurement families and shrunk within the family. Confidence must
depend on independent families, not on the raw number of metrics.

### 5. Shared and task-specific states are represented as competing columns

When task-specific states are present, the Agent workspace suppresses the
overall copy to avoid duplicate voting. That prevents double counting, but it
also removes the useful distinction between a shared subject-level state and a
task residual. Multi-task fusion should retain both with an explicit hierarchy:

`state_subject = shared_state + task_residual`

The residual is shrunk toward zero for small tasks. The shared and residual
terms must never be counted as two independent votes.

### 6. The supervised predictors return probabilities without a sufficient explanation packet

The Agent currently receives module A and module B class-probability vectors.
It does not receive the state contributions, uncertainty, out-of-distribution
score, route-specific calibration status or reasons for disagreement. It cannot
perform a principled adjudication from two probability dictionaries.

### 7. Agent state revision is not numerically executable

The current Agent can downweight, invalidate or mark a StateCard unavailable.
After this action, cached supervised outputs correctly become stale, but the
state aggregation and predictor are not replayed. The final decision therefore
cannot show how the revision changed the probability. This is the largest open
architectural gap.

### 8. Agent authority is currently all-or-nothing

In Agent-led mode, the Agent owns the final class after mandatory checks. In the
legacy path, a bounded correction policy can make the Agent almost irrelevant.
Neither extreme is adequate. Mandatory consultation does not define what to do
when the Agent and learned experts disagree.

### 9. The Agent ordinal score is not a calibrated probability

The Agent emits integers from 0 to 4. These scores are useful as structured
evidence likelihoods, but they cannot be reported as disease probabilities or
compared with probability AUROC without a version-specific development
calibrator.

### 10. The Agent does not inspect the waveform

The segment tool returns stored measurements and timing metadata. It explicitly
does not load the waveform. This is a valid current boundary, but the method
must not describe it as Agent audio understanding. Future direct audio tools
require a separate privacy, provenance and validation contract.

## Four observation channels and one independent target route

| Observation family | Current datasets | Strong evidence | Main restrictions |
| --- | --- | --- | --- |
| Structured clinical interview | IAEAV | Patient pauses, output efficiency, lexical states, repair, patient turn share and response burden | Requires reliable roles; interviewer gaps cannot become patient pauses; prompt style is a confound |
| Picture description / structured single task | ADReSS 2020, ADReSSo diagnosis, NCMMSC long track | Pauses, output efficiency, lexical retrieval/diversity, repairs; content units when a validated scorer exists | No interaction state without roles; S09/S10/S14 need task and language validated scorers |
| Structured cognitive multi-task | PREPARE, PROCESS-2, DementiaBank Pitt | Task-specific pause/output/lexical/repair states; task performance when a scoring key exists | Cookie, recall, fluency and reading must not be averaged before fusion |
| Spontaneous or non-standard speech | TAUKADIAL, DementiaNet public speech | Broad fluency, output, lexical and selected prosodic evidence | No standard task score; public media has device, editing and context confounds; public speech is a stress test, not a primary training source |

`ADReSSo_2021_progression` uses a longitudinal target route. It requires paired
visits, interval metadata and within-person state deltas. It should not share the
cross-sectional diagnostic head merely because the recordings are speech.

## Revised six-stage architecture

### Stage 1. Data governance, routing and quality contract

Input: audio, transcript, task, language, role, visit and source metadata.

Actions:

1. Freeze subject-disjoint train, development/calibration and test manifests.
2. Route the case by observation family and target endpoint independently.
3. Preserve task boundaries and patient/interviewer roles.
4. Run acquisition, VAD, diarization, ASR, alignment and duration checks.
5. Produce an `observability_mask` for every state and task instance.
6. Produce case-level OOD indicators for language, task, device and duration.

Output: a label-free routed case manifest. Test labels remain outside inference.

### Stage 2. MetricEvidence compiler

Every metric becomes a typed evidence object. The minimum contract is:

- stable evidence, metric, state, subject, session, task and segment IDs;
- value, unit, method version and source modality;
- training-fold reference median, scale, sample size and artifact hash;
- abnormal direction and direction provenance;
- observability and the reason for unavailable status;
- reliability components: source, role, alignment, ASR, reference support and
  measurement stability;
- potential, observed and ruled-out confounds as separate fields;
- inference permission and report permission as separate fields;
- `consumed_by_supervised` and `incremental_for_agent` flags.

The Agent cannot promote an unavailable metric, change its direction, or grant
report permission. It may only request a validated remeasurement or reduce its
use through an evidence revision.

### Stage 3. Task-conditioned cognitive StateGraph

The shared state ontology remains S01-S14. Each observation is represented as:

`Sxx@task = shared Sxx + task residual + reliability + trajectory`

Within-state fusion follows this order:

1. remove unavailable measurements;
2. group correlated metrics into independent measurement families;
3. compute a robust family score;
4. combine families with reliability-constrained weights;
5. retain support, counterevidence, confounds and segment trace;
6. produce shared and task-residual views without duplicate votes.

The first implementation should use fixed clinical weights plus hierarchical
shrinkage. Learned within-state weights are enabled only when the inner folds
show stable improvement and sufficient subjects per task and language.

Required additions to the current state set are task-validated semantic tools:

- picture-content units and omissions;
- semantic-fluency clustering and switching;
- recall omissions, intrusions and ordering;
- narrative relevance and topic drift with language-specific validation.

Without these tools, the Agent mainly sees pause and output evidence and cannot
be expected to improve the MCI/AD boundary consistently.

### Stage 4. Supervised module A: task-conditioned statistical expert

Module A learns statistical class boundaries. It does not write the report and
does not decide whether evidence is medically reportable.

Inputs:

- shared and task-residual StateCards;
- observability masks and reliability components;
- capped audio/text representations;
- task and language adapters;
- QC only as a reliability modifier, never as disease evidence.

Outputs must be an explanation packet, not only probabilities:

- calibrated class prior;
- state and branch contributions;
- uncertainty and fold disagreement;
- OOD indicators;
- applicability status;
- artifact and evidence-snapshot hashes.

Model complexity is data-dependent. Small cohorts use regularized linear or
ordinal models. Larger cohorts may use a low-capacity hierarchical gate. Frozen
or fine-tuned encoders are a separate ablation from the cognitive framework.

### Stage 5. Single Agent evidence adjudication and replay

The Agent receives the StateGraph, MetricEvidence, transcript, segment tools,
quality observations and the AD skill package. It follows a fixed sequence:

1. inspect quality and task observability;
2. inspect high-priority states and counterevidence;
3. inspect the transcript and available segment measurements;
4. record a blind evidence hypothesis;
5. inspect module A's explanation packet;
6. resolve disagreement by citing evidence IDs;
7. retain, downweight, invalidate or mark unavailable specific evidence/states;
8. request numerical replay;
9. inspect the replayed result and submit a candidate judgment.

Replay is mandatory after any accepted revision:

`reviewed MetricEvidence -> rebuilt StateGraph -> frozen module A -> new packet`

The Agent is allowed to decide evidence applicability and clinical coherence.
It is not allowed to edit raw values, invent a measurement, expose labels, set a
probability directly, or convert QC/model-only features into medical evidence.

### Stage 6. Module B calibration, locked decision and traceable report

Module B is a low-capacity conditional arbitration and calibration layer fitted
only on cross-fitted development outputs. It receives:

- module A prior before and after replay;
- Agent ordinal evidence scores;
- Agent/model agreement pattern;
- revision type and cited evidence;
- evidence coverage, reliability and confound burden;
- task, language and OOD status.

Its purpose is not to learn a global Agent percentage. It learns whether the
Agent supplied validated incremental information for a particular route and
action type. If the Agent only reinterprets evidence already consumed by module
A, the primary effect must occur through replay, not a second additive vote.

The final run locks one evidence revision, one model packet, one calibrated
decision and one report. The report trace is:

`claim -> StateCard revision -> MetricEvidence -> task/segment -> source asset`

Benchmark mode always emits one class and calibrated probabilities. Clinical
mode emits the most likely class, evidence sufficiency, uncertainty, limitations
and repeat/referral advice; insufficient evidence does not erase the most likely
research classification.

## Responsibility boundary

| Decision | Deterministic rules | Supervised modules | Agent |
| --- | --- | --- | --- |
| Is a state observable in this task? | Owns the base rule | No | May identify a rule mismatch but cannot promote it |
| Is this measurement technically reliable? | Defines components and minimum checks | May calibrate reliability from training data | May reduce or challenge reliability with cited evidence |
| What statistical class boundary fits the training data? | Enforces split/provenance | Owns | Reviews but does not fabricate a new boundary |
| Are support, counterevidence and task conflicts clinically coherent? | Enforces schema and citation validity | Supplies contributions | Owns the review |
| Does a state revision change risk? | Requires replay | Recomputes the probability | Initiates and explains the revision |
| What is the calibrated probability? | Enforces version binding | Owns calibration | Supplies ordinal evidence, not probability |
| What can appear in the clinical report? | Owns permissions | No | Selects among allowed claims and explains limitations |

## Conditional authority policy

There is no fixed 60/40 or 80/20 split. The following conditions determine how
much an Agent action can change the decision:

| Case condition | Supervised authority | Agent authority | Required action |
| --- | --- | --- | --- |
| In distribution, high evidence coverage, experts agree | High | Review and explanation; override requires strong contradictory evidence | Preserve prior unless replayed evidence changes it |
| Experts disagree but evidence is rich | Moderate | High evidence-review authority | Inspect conflict, revise if supported, replay |
| OOD language/task but interpretable evidence is available | Reduced | Higher review authority, limited probability authority | Use route-specific calibration and widen uncertainty |
| Poor audio/ASR/role coverage | Low for affected branches | May withhold or request repeat collection, not invent a class signal | Clinical uncertainty; benchmark uses predeclared fallback |
| Agent revises a state | Stale until replay | Owns the proposed revision, not its numerical effect | Recompute before finalization |
| Only model-only acoustic evidence supports risk | Predictive use under a cap | Cannot cite it as clinical support | Report the lack of interpretable support |

Future stronger models gain authority only after the same model/skill/tool version
shows positive cross-fitted value for the relevant channel, language and action
type. Model size alone does not change the authority policy.

## Current empirical interpretation

The evidence framework itself has shown useful signal. In the recorded matched-
encoder ablation, adding MetricEvidence, StateCards and task routing improved
accuracy by 0.0752 and macro AUROC by about 0.0601. This supports preserving the
cognitive evidence layer.

The independent PREPARE result has not surpassed SpeechCARE. The recorded
ADvoice result has accuracy 0.6723 and micro AUROC 0.8462, compared with the
SpeechCARE paper's 0.7211 micro F1 and 0.8683 micro AUROC. SpeechCARE jointly
fine-tunes mHuBERT and mGTE and learns adaptive modality gates on PREPARE; the
current independent ADvoice path uses a different training protocol.

The latest 12-case, four-channel Agent-led pilot is a functional diagnostic, not
a performance estimate. Requiring blind hypothesis plus advisor review raised
Agent accuracy from 4/12 to 6/12, while the frozen supervised system obtained
8/12 on the same tiny sample. This shows that model consultation alone does not
solve conflict adjudication. It does not establish channel-level accuracy or a
SpeechCARE comparison.

## Evaluation required before broad execution

### Layer A: predictive performance

- accuracy, macro/micro F1, class precision and recall;
- macro, micro and weighted AUROC/AUPRC;
- log loss, Brier score and expected calibration error;
- confusion matrices, task/language/channel subgroups;
- repeated seeds and paired confidence intervals;
- coverage-risk curves for clinical abstention mode.

### Layer B: evidence and report validity

- observability-rule violations;
- report-permission violations;
- trace completeness and segment faithfulness;
- counterevidence inspection rate;
- stale-advisor and stale-revision failures;
- revision acceptance, replay success and revision utility;
- unsupported disease/stage/causal claims;
- decision consistency across repeated Agent runs;
- clinician review only when actual clinician ratings are available.

### Required ablations

1. Same encoders, supervised module A only.
2. Direct Agent with transcript, without MetricEvidence/StateCards.
3. Agent with evidence graph, without learned advisor outputs.
4. Agent with advisor outputs but without replay.
5. Full evidence review, replay and conditional module B.
6. Full system without task-specific states.
7. Full system without semantic task tools.

For SpeechCARE, only the complete official PREPARE cohort with the same endpoint,
official split, declared preprocessing, repeated seeds and confidence intervals
supports a matched claim. Accuracy alone is not enough. The same comparison must
also report MCI recall, AD recall, calibration and subgroup performance.

## Implementation order for the next engineering Agent

1. Freeze and test separate observation-route and target-route schemas.
2. Extend MetricEvidence with component reliability, direct provenance and
   evidence-consumption fields.
3. Add the task-by-state observability registry and hierarchical shared/task
   StateGraph representation.
4. Add correlation-family shrinkage inside each state.
5. Change module A output from probabilities to a versioned explanation packet.
6. Implement typed evidence revisions and deterministic state/model replay.
7. Replace the global Agent correction with cross-fitted conditional module B.
8. Add task-specific semantic tools, beginning with PREPARE task types.
9. Run small prespecified pilots for interview, picture, multi-task and
   spontaneous/public channels.
10. Run the complete PREPARE matched protocol only after the replay and
    calibration gates pass; expand to the remaining datasets afterward.

## Promotion gates

Do not promote a candidate merely because one accuracy number improves. Promotion
requires all of the following:

- no split, reference or identity leakage;
- no regression in calibration beyond the prespecified margin;
- stable gain or noninferiority across folds and seeds;
- no clinically important class-recall collapse;
- no increase in invalid evidence or report-permission violations;
- Agent revisions are replayed and auditable;
- the locked report cites the exact evidence revision that produced the final
  probability.

## Source files reviewed

- `src/advoice/evidence.py`
- `src/advoice/states.py`
- `src/advoice/models.py`
- `src/advoice/condition_c.py`
- `src/advoice/diagnostic_agent.py`
- `src/advoice/agent_led.py`
- `src/advoice/pipeline.py`
- `configs/datasets/*.yaml`
- `configs/channels/*.yaml`
- `configs/metrics/audio_metrics.yaml`
- `configs/states/audio_states.yaml`
- `skills/ad_evidence_diagnostic/*.md`
- `docs/SYSTEM_REVIEW.md`
- `docs/PREPARE_SPEECHCARE_GAP_ROOT_CAUSE_2026-09-03.md`

External comparator: SpeechCARE, npj Digital Medicine (2025),
https://www.nature.com/articles/s41746-025-02026-x
