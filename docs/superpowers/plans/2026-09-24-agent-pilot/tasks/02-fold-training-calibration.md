# T2: Implement Fold-Safe Learning and Joint Calibration

Model: gpt-5.6-sol; reasoning: high.
Depends on: T0.
Read START_HERE.md and CONTRACTS.md first. All repository-relative paths resolve from the assigned worktree root.
Write ownership: `src/advoice/pilot/learning.py`, `tests/test_pilot_learning.py`.
All source code and public UI text must be English. Do not launch unrelated work.

## Inputs
Shared contracts, existing authority_study_dataset / condition_c_advisor / authority_joint_fusion, existing trained model adapters and feature pipeline. Obtain exact symbols through graph discovery. Read current source, not historical comments alone.

## Steps
- [ ] Identify the actual full base predictor and state replay predictor. Record classes, feature signatures and training recipe. Keep their identities distinct; do not accidentally compare a weakened state-only model against a full Agent model.
- [ ] Wrap existing low-capacity fitting with explicit fit_ids and excluded_ids. All imputation, reference fitting, state transforms, feature selection and model fits stay within fit_ids.
- [ ] Implement fit_fold/predict_fold to emit paired base prediction, same-fold evidence context, and replay model. Never load old ours_predictions.csv as new OOF training data.
- [ ] Emit OOF provenance for every development patient exactly once. Require observed fit IDs to exclude that patient's entire identity group.
- [ ] Implement fixed regularized binary/two-head calibration from CONTRACTS.md; B_raw, calibrated B, J-A, J-S, J-AS. Freeze score contrast definitions and feature scaling.
- [ ] Combine three-class heads as P(HC)=1-p_imp, P(MCI)=p_imp*(1-p_AD_given_imp), P(AD)=p_imp*p_AD_given_imp. Train staging head only on labeled MCI/AD development rows.
- [ ] Save optimizer success, objective, coefficient/scaler values, head fit IDs, class order and seed. Failure or absent training class gives a declared non-estimable head/fallback, never invented coefficients.
- [ ] Missing Agent/replay inputs use the declared B fallback. Do not treat missing values as confident healthy evidence. Produce matched-fit-row sensitivity calibrators for comparison.
- [ ] Final base/state models refit on full development ONLY after OOF collection; Module B remains trained on OOF rows.
- [ ] Expose pure scoring functions so all conflict combinations can be tested without API calls.

## Tests
- [ ] Mutating holdout labels does not alter references, coefficients or predictions.
- [ ] A deliberately leaky fold is rejected.
- [ ] Coefficient prior (1,0,0,0) reconstructs the base on raw base log-odds.
- [ ] Uninformative Agent does not receive a forced minimum positive coefficient.
- [ ] Agent up / state down, Agent down / state up, uncertain base / weak Agent, binary absent MCI and invalid second pass are handled deterministically.
- [ ] Class permutation round-trip, probability sum=1, finite clipping, constant features and optimizer failure.
- [ ] State replay zero delta is neutral; no diagnosis relabeling to improve a metric.
- [ ] Serialize/load yields identical predictions within 1e-10.

## Acceptance
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q tests/test_pilot_learning.py tests/test_pilot_contracts.py
```
Expected: leakage and numerical tests pass; deterministic synthetic training completes. No actual cohort fitting or paid inference during this coding task. Do not tune the protocol from synthetic desired accuracies.

## Required completion response
Write .local/pilot-handoff/T2/WORKER_RESULT.md with: base SHA, final SHA, changed files, exact test commands/results, artifact paths, remaining findings, schema/config changes requested, paid calls and tokens (zero for code-only tasks). Return this path and a summary under 400 words. Do not paste patient data or the entire code diff into coordinator chat.
