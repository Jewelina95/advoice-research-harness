# ADvoice system review

Review date: 2026-09-17. Scope: the canonical repository, historical 8.27/9.2
artifacts, training/evaluation code, and the executable diagnostic-Agent contract.

## Findings before conclusions

| Priority | Finding | Action / status |
| --- | --- | --- |
| P1 | Inner CV reused reference states fitted beyond the inner fitting partition | Rebuild states and prototypes inside inner fitting folds |
| P1 | Fusion reliability did not follow the fold of its out-of-fold predictions | Fold-specific reliability now accompanies those predictions |
| P1 | Exported development priors used a temperature fitted with their own labels | Use outer-fitting-partition calibration for those priors |
| P1 | Progression calibration assumed a literal HC reference class | Use the configured reference class; regression covers 35 training subjects |
| P1 | Embedded StateCard metrics were visible but absent from the Agent ID registry | Register inference-visible embedded IDs; keep permission checks |
| P1 | An invalidated state could still support an accepted risk correction | Reject citations to invalidated states and attached evidence |
| P1 | Development routing used raw scores while inference routed calibrated scores | Route calibrated OOF scores before correction-strength selection |
| P1 | A batch response could contain another batch's case or duplicate decisions | Request-local whitelist and duplicate rejection |
| P1 | Potential confound tags were interpreted as assessed case-level findings | Separate potential tags; unknown assessments cannot authorize correction |
| P1 | Correction selection did not guard probability-quality deterioration | Add prespecified log-loss and Brier noninferiority checks; reject missing-class calibration |
| P1 | Duplicate participant rows could inflate evaluation counts | Reject duplicate/null subject IDs at artifact boundaries |
| P1 | PREPARE comparison accepted an arbitrary prediction file as the official cohort | Require the frozen 412-subject ID set and valid probabilities |
| P1 | State edits still do not reexecute the state-based supervised scorer | Open architectural gap; do not claim executable cognitive-state revision |
| P1 | Generic model/expert selection is not fully independent of the calibration OOF pool | Dedicated held-out calibration needed before strong calibration claims |
| P2 | Full batch execution depended on beating a published result | Decouple scientific benchmark claims from dataset execution |
| P2 | Code, data and outputs were coupled to a dated local directory | Add explicit private workspace/raw-data/model-config paths |
| P2 | Disabled processed Agent runs could still invoke report scoring | Disable scoring with the Agent; no hidden API call |

## What the system actually computes

1. Dataset adapters identify subject, task, language, recording and available speaker roles.
2. Acoustic, text, dialogue and task measurements become evidence objects with
   provenance, reference scope, reliability, applicability and report permissions.
3. Training-fold reference statistics map measurements into shared and task-specific
   cognitive states. Missingness must not become a normal-state observation.
4. Supervised experts produce out-of-fold text/audio/state predictions; stacking or
   a dynamic gate combines them. The chosen model and temperatures are recorded.
5. The single diagnostic Agent receives a prior-blind evidence workspace. It returns
   cited evidence scores and state-review actions. A development-fitted correction
   policy decides whether those scores may change the supervised probabilities.
6. Invalid or unsupported Agent corrections fall back to the supervised prior.
   Reporting uses the locked prediction and separately authorized report evidence.

This is not agent fine-tuning or LLM post-training. It is supervised predictor and
calibrator training plus constrained LLM inference. It is also not yet a fully
executable loop in which Agent state edits regenerate the supervised prediction.
Rejecting a citation protects the correction path; it does not remove that feature
from a prior that was already computed.

## Evidence against SpeechCARE

The checked historical independent ADvoice PREPARE run has accuracy/micro F1
0.672330 and micro AUROC 0.846162. The paper benchmark used by this repository is
0.7211 and 0.8683 respectively. Thus the accuracy gap is 4.88 percentage points;
the independent system has not surpassed SpeechCARE.

The historical extension using released SpeechCARE predictions plus cognition
has accuracy 0.735437. It is a different experiment, not an independently trained
ADvoice model. Comparing that hybrid with a paper mean cannot establish that this
repository's standalone method is superior.

The same 412 test IDs do not establish identical training input, preprocessing,
training budget or seed protocol. The current record also contains multiple
PREPARE training entry points. Generic CLI training, representation-fusion
experiments, and released-output extensions must retain separate identities.

The existing official test outcomes have been inspected repeatedly. New runs on
them remain retrospective, even when candidate selection is now confined to
development data. Confirmatory superiority needs a newly locked external cohort.

## Current empirical checks

The original 133-test suite passed before edits. New regressions exercise the
specific failures above, rather than just checking successful execution.

Cross-task pilots use explicitly frozen historical preprocessing, retrain the
current supervised condition C, and disable live LLM calls. B1/B2 remain frozen
historical comparators. These are engineering and predictive pilot checks, not
fresh full-agent clinical validations. The generated review report reads actual
pilot CSV files and links to the preserved Layer A and Layer B reports.

The registry contains ten active task configurations, not ten independent cohorts:
ADReSSo diagnosis and progression are separate tasks, and DementiaBank-derived
cohorts may share provenance. Every dataset is processed separately. NCMMSC pilots
use long recordings only; no six-second dataset is admitted.

## Remaining release conditions

1. Validate the full expert/model-selection boundary with an independent calibration
   holdout, not merely cross-fitted feature scaling.
2. Implement and validate a producer for the new case-level confound assessment
   interface. Potential tags are now separated, but a schema alone is not an
   observed assessment; ordinary current workspaces cannot authorize corrections.
3. Implement reviewed state edits as a typed, replayable intervention on a frozen
   state scorer; evaluate with the same encoders and prespecified development set.
4. Verify task labels and source identities for datasets with incomplete metadata.
5. Audit Layer B endpoints individually: label-informed state interventions are
   oracle sensitivity checks, not evidence of deployable clinical correction.
6. Establish Agent benefit with live calls on locked workspaces after the relevant
   development gate passes. Unit tests and report-quality scores are not substitutes.
7. Confirm upstream redistribution rights for protocol/transcript reference tables
   before a publication release; never add private participant data to Git.

Do not broaden a long, expensive batch merely because its code executes. Complete
the integrity checks and bounded pilots first, retain failed candidates, and then
freeze the broader evaluation protocol.

## Primary comparator sources

- [SpeechCARE paper](https://www.nature.com/articles/s41746-025-02026-x)
- [SpeechCARE released implementation](https://github.com/SpeechCARE/SpeechCARE-NIA-Phase2)
- Local comparison artifact: `.local/review/reports/current_speechcare_comparison.json`.
- Private experiment inventory: `.local/review/reports/inventory/inventory.json`.
- Repository management: [Research workspace](RESEARCH_WORKSPACE.md).

## Bounded experiment outcome

Eight prespecified PREPARE frozen-representation heads were trained using the
public 1,295/327 training/validation split. At matched regularization, adding
cognitive states to OOF stacking improved validation accuracy from 0.669725 to
0.685015 (five additional correct cases). Adding the same states by concatenation
decreased accuracy. Thus richer evidence does not automatically improve a model.

The selected head was locked before one retrospective official-test evaluation.
It correctly classified 274/412 subjects: accuracy 0.665049, macro F1 0.558180,
macro AUROC 0.804681, and micro AUROC 0.846914. It is not promoted. Released
SpeechCARE raw and bias-mitigated predictions have accuracy 0.706311 and 0.737864
on those same subjects. These individual checkpoints are distinct from the
paper's ten-run mean, 0.7211. Same cohort does not establish the same input,
encoder-training or repetition protocol.

The selected model predicts HC for 74/132 ADRD cases and 28/51 MCI cases. This is
an observed error pattern, not proof that any one encoder or component caused it.
No additional candidate was evaluated after seeing these test results. The test
evaluation verifies model/input hashes and performs zero fitting operations.

Four separate processed-input pilots completed: ADReSS2020 internal holdout,
ADReSSo progression, NCMMSC long recordings and PublicFigures. Their reports
retain Layer A and Layer B. They do not validate live Agent correction, and weak
progression/public-speech discrimination prevents a general efficacy claim.

## Next controlled upgrade

1. Hold the public test fixed as retrospective reporting only. Use a development
   split for model choices and a disjoint calibration partition for correction
   strength. Persist subject/source identities and all transformation fit scopes.
2. Build a replayable state intervention contract: invalidate affected evidence,
   reaggregate its state, rerun the frozen state head, then compare against the
   unchanged input experts. Unsupported interventions must abstain explicitly,
   not be described as removal from an already-computed prior.
3. Validate case-level quality assessments from actual tool observations, with
   scope and provenance. Do not infer absence of confounds from absent tags.
4. Keep the same encoders while evaluating the intervention mechanism versus
   supervised-only and prior-blind Agent conditions. Separately test temporal
   representations and task-specific states on development data. This separates
   evidence-framework benefit from encoder upgrades.
5. Require stable classwise recall, calibration and paired performance across
   prespecified seeds before promoting a candidate. Expand to all ten registered
   tasks only after representative task pilots pass their declared criteria.
6. Use a newly reserved external cohort for confirmatory superiority. PARLO is
   a possible German multicentre source but requires approved DementiaBank
   access; it has not been downloaded or added to any result here.

Primary external-cohort page: https://talkbank.org/dementia/access/German/PARLO.html
