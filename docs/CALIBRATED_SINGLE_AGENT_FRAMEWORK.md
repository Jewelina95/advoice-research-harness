# Calibrated single-Agent evidence framework

This document defines the maintained evaluation architecture. It is deliberately
different from the experimental fully Agent-led tool loop.

## Execution contract

1. Route each case by dataset, task family, language, available modality and
   speaker role. Subject isolation and task boundaries are fixed before fitting.
2. Produce acoustic, language, dialogue and task measurements. Convert each
   permitted measurement into typed `MetricEvidence` with source, direction,
   reference scope, reliability, confounds, task and segment provenance.
3. Aggregate non-duplicated evidence into shared and task-specific `StateCards`.
   Missing or task-inobservable states remain unavailable rather than imputed as
   normal.
4. Fit the supervised reference using training-fold data and generate out-of-fold
   development probabilities. Evaluation labels never enter the workspace.
5. Make one structured, prior-blind Agent request. The Agent reads the transcript,
   evidence graph, state cards, counterevidence, quality constraints and versioned
   AD skill. It returns cited state actions and ordinal class evidence.
6. Validate evidence identifiers and permissions. Replay accepted state actions
   deterministically to obtain the pre/post state change.
7. Fuse only with strengths selected on subject-disjoint development cases. For
   HC/MCI/AD, screening and staging are separate decisions. State-action and blind
   screening are alternative routes from the same Agent call, never two votes.
8. Lock the prediction and its trace. Report generation is downstream and cannot
   alter classes, probabilities, evidence selection or release permissions.

## Calibration contract

`agent_correction_calibration.json` must have
`selection_status=validated_joint_gain`. Its screening strength controls the
blind ordinal screening route; its staging strength controls only the
conditional MCI-versus-AD route. Replayed state deltas use a different scale and
remain zero unless the artifact separately records
`state_selection_status=validated_joint_gain` and `selected_state_strength`.
The artifact is selected from development/OOF predictions under macro-F1 gain
plus AUROC, log-loss and Brier non-inferiority constraints. Missing, invalid or
unsuccessful calibration produces exact supervised parity.

Manual nonzero strengths remain available only for controlled method debugging.
They are not publication results and must not be selected from evaluation data.

## Stage semantics

The class vocabulary is route-specific. Current HC/MCI/AD datasets support a
screening boundary and a coarse cognitive-status boundary; speech alone does not
establish Alzheimer pathology. A dataset may later expose SCD or dementia
severity labels, and the generic fusion preserves those labels, but each endpoint
requires its own subject-disjoint calibration and evidence-observability policy.

## Cost boundary

The default path makes one provider call per uncached case. Compact aliases are
expanded back to full evidence IDs after validation. Report-only policy documents
and clinician prose are excluded from the decision request. Provider, model,
skill, policy content, review mode and schema are part of the cache identity.
