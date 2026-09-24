# Frozen Implementation Contracts

Status: proposed implementation contracts; no production code is changed by these documents.
Owner: T0. Only T0 changes this file or the shared Python contract module.

## Reuse boundary

Read existing AuthorityStudyDataset, FrozenConditionCAdvisor, authority_joint_fusion, authority_review_runtime, agent_runtime, conditional_authority, routing and the state/metric YAMLs before writing adapters. Use graph discovery, then source snippets; graph metadata is not a substitute for the current dirty source.

Do not replace the whole system. Existing historical prediction CSVs can be descriptive comparators, NOT training OOF features for a new split. Preserve the existing validated encoder, metric and state extraction implementation. No new unvalidated states, lexicons or clinical stages to fill a diagram. A state-only replay predictor is not the full frozen Condition C model: store base_model_id and replay_model_id separately.

## Typed records

T0 implements these versioned records in src/advoice/pilot/contracts.py, using existing repository data classes for the underlying evidence/state bodies.

- SubjectRow: dataset_id, pseudonymous subject_id, source_group_id, channel, language, task_ids, role, partition, fold_id, raw_hashes, source_version. Partition is exactly engineering_canary / development / holdout / stress; these sets are mutually exclusive within a run. No label in inference serialization.
- LabelRow (separate file and reader): dataset_id, subject_id, target_name, class_label, label_source, label_mapping_version.
- EvidenceSnapshot: case_id, state_version, snapshot_hash, reference_fit_id, source segment IDs/timestamps/roles, MetricEvidence objects, StateCards, observability reasons, confounds, skill_hash, extractor_hash.
- AgentAssessment: case_id, snapshot_hash, status, supported_class_scores, claim_citations, state_assessments, bounded_operations, model_id, usage_id. Scores are ordinal 0..4 support, not probabilities.
- ReplayResult: parent_hash, child_hash, accepted_operations, rejected_operations, invalidated_ids, recomputed_ids, before_state_scores, after_state_scores, replay_model_id.
- FusionRow: subject_id, dataset_id, partition, fold_id, base_fit_id, reference_fit_id, base_probabilities, agent_v0_scores, agent_v1_scores, replay_delta, validity, snapshot_hashes. No truth until training/score join.
- PredictionRow: subject_id, arm, class_order, class_probabilities, predicted_class, status, fallback_reason, calibrator_id, source_trace_ids.
- UsageRow: request_id, cache_key, model_id, attempt, input_tokens, output_tokens, reasoning_tokens, reported_cost, wall_seconds, response_status. Missing usage/cost is null with reason, never zero.

Semantic states are observed / not_observable / missing / conflicted, not a shared numeric zero. Trace IDs are immutable and content/version bound. Labels and diagnosis-bearing filenames are excluded from ALL provider inputs and tool responses, not merely the top-level prompt.

## Planned public interfaces

These are new interfaces to be implemented, not commands/functions claimed to exist today.

- data.build_pilot_manifest(config, source_inventory) -> manifest + exclusions + audit
- data.make_group_folds(development_manifest, labels, seed, n_splits=5) -> folds
- learning.fit_fold(inputs, fit_ids, validation_ids, frozen_config) -> fold_artifact
- learning.predict_fold(fold_artifact, case_inputs) -> base + state_snapshot + replay_context
- learning.fit_joint_calibrators(oof_rows, development_labels, config) -> calibrators
- runtime.assess_and_replay(snapshot, provider, executor, budget, cache) -> assessments + replay + usage
- learning.predict_joint(fusion_row, calibrators) -> predictions for B_raw/B/J-A/J-S/J-AS
- reporting.render_run(run_dir, output_dir) -> HTML paths
- runner.run_stage(stage, resolved_config, run_dir) -> stage_manifest

Arguments must be typed with the shared records/protocols, not arbitrary positional dictionaries passed between workers. Existing model/reader dependencies are injected adapters. Tests use no-network fakes.

## Training and prediction semantics

Frozen encoder outputs may be reused only with verified training/source provenance. Fit imputation, normalization, feature selection, reference ranges, state-dependent transforms and supervised heads ONLY on the current fold fit IDs.

For each development patient, produce exactly one OOF prediction and its SAME-fold evidence/reference context. Train two diagnostic binary heads where labels permit: impaired vs HC; AD vs MCI conditional on impaired. Binary HC/AD datasets use one binary head, not fabricated MCI targets. Progression is not a diagnosis head. AD stage output is disabled without corresponding labels.

Mandatory stage order: complete fold-fitted OOF rows and bound Agent assessments -> fit all calibrators on OOF -> freeze calibrator hashes -> refit base/state models on full development -> predict holdout. Final-refit predictions must never flow back into calibrator fitting. A calibrator input manifest containing final-fit IDs is rejected.

Runner capability manifests distinguish development labels from holdout labels: split/fit-oof/fit-fusion may read only their authorized development labels, while score alone reads holdout/stress truth after prediction hashes freeze. assess-oof/predict-holdout/provider tools receive no label-reader capability. A provider-boundary validator rejects LabelRow objects, truth-bearing paths and base predictions, checks reference_fit_id against the fold manifest, and stops on violations. Labels are not secured merely by telling the model to ignore them.

Base and replay state predictor configurations must be declared separately if they differ. Replay uses the same fold-fitted state model before/after; if either score is unavailable, mark delta unavailable rather than fitting a convenient replacement on the test case.

For an ordinal vector s, fix contrasts: impaired=(s_MCI+s_AD)/2-s_HC; stage=s_AD-s_MCI; binary=s_AD-s_HC. These are score features, not log-likelihood evidence from independent observations. Nonexistent classes are not imputed.

Clip base/state probabilities to [1e-6,1-1e-6] for log odds. A head uses [base log odds, agent contrast, replay log-odds delta] plus intercept. Preserve base log-odds scale; scale other features on that head's training rows. Minimize mean cross-entropy + lambda * squared distance to (1,0,0,0), lambda=1; coefficients except intercept nonnegative. No hyperparameter search.

B gets the same OOF slope/intercept calibration budget. J-A uses v0 Agent score; J-S uses state replay only; J-AS uses v1 score after accepted meaningful changes, otherwise v0. Fit each arm independently on its own valid OOF inputs; report a matched-fit-row sensitivity comparison. Fallback to frozen calibrated B for invalid/missing required inputs. All assigned heldout subjects remain in the denominator. Save per-head valid fit counts and refusal reasons.

No valid two-class training support / failed optimizer / class-order mismatch => block that head and record explicit baseline fallback or unestimable result. Never silently reuse another dataset's calibrator. Evaluate generalization only on the holdout, not on rows fitting Module B.

## Agent authority and traceback

Agent reads role/task-aware transcript spans, evidence, state, confounds and applicable skills; no base class or probability. Shared evidence is legal; consumed_by_supervised remains provenance metadata, not an unconditional zero gate for this path. Retain old strict independent-evidence mode for historical reproduction.

Separate evidence legality, measurement reliability, and learned prediction contribution. High base uncertainty alone cannot grant authority. Do not force a positive Agent coefficient to make the Agent appear useful.

Allowed bounded operations: request an existing deterministic remeasurement, correct a source-backed role/span association, flag an unsupported interpretation or a confound. Never invent raw metric values, diagnosis facts or a new clinical reference. Max 3 proposed operations, max 1 accepted repair batch. The executor, not the Agent, recomputes affected descendants.

First assessment is bound to v0. Meaningful accepted change invalidates dependent scores; at most one fresh v1 assessment with no operations. A failed required second assessment does not resurrect stale v0 for J-AS. J-S can remain evaluable; J-AS falls back to B with reason.

Trace result -> fitted head/Agent assessment -> state version -> evidence measurement -> task/role/span/audio hash. Provenance trace is not hidden chain-of-thought or proof of causal clinical validity. Opaque embeddings are explicitly marked, not falsely explained by state cards. Frozen-base branch intervention and end-to-end raw-input intervention are distinct experiment types.

## Cache and cost

Cache key includes input hash, fold reference/model hashes, task/role, extractor, state schema, skill/prompt hash, exact provider model ID/settings, assessment round, parent snapshot and operations. No reuse on any mismatch. Log hit/miss reason.

A subject in one core analytical pipeline has max 2 semantic Agent calls. In the separately budgeted model comparison, the cap is per (subject, model, representation) cell: structured max2, flat max1; total max6 per comparison subject across two models. No cross-cell cache reuse unless every identity/version/input key matches. Transport retries max 1 per request, idempotent where supported; count attempts/tokens even on failure. No automatic malformed-JSON model-repair loop. Missing responses are failures, not silently filled fake outputs.

Global concurrency 2 initially; one training process per accelerator. Max provider request wall time 180 seconds; 2 consecutive transport failures pauses the provider queue. Identity leak, untraceable accepted patch, wrong snapshot or class mapping stops the run immediately. Accuracy errors do not trigger an early stop or sample substitution.

Core semantic call cap 742; 24-subject paired model comparison cap 144 (96 initial + at most 48 structured second-pass); canary cap 24 separately. Conservative total semantic cap 910, transport attempts separately bounded and included in cost.

Resolve a concrete provider model, pricing source/date and input/output/reasoning token limits before a paid call. Model-selection aliases are not automatically valid API model identifiers. Reuse the working provider, do not assume Codex subagent model names are available through the public API.
