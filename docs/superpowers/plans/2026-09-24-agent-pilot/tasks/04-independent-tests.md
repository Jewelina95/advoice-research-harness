# T4: Independent Integration and Intervention Tests

Model: gpt-5.6-terra; reasoning: high.
Depends on: T1, T2, T3.
Read START_HERE.md and CONTRACTS.md first. All repository-relative paths resolve from the assigned worktree root.
Write ownership: `tests/test_pilot_integration.py`, `tests/test_pilot_interventions.py`, `tests/fixtures/pilot/synthetic_cases.json`.
All source code and public UI text must be English. Do not launch unrelated work.

## Inputs
Integrated T1/T2/T3 commit, contracts, supplied fixtures and current tests. You are a test owner, not a production-code author. Return defects to their owners.

## Steps
- [ ] Construct synthetic fixtures with hand-calculated expected signs and fallback behavior, not snapshots copied from implementation output.
- [ ] Cover the Cartesian conflict matrix: base certainty low/high x Agent evidence invalid/weak/valid x Agent contrast +/- x state delta +/-/zero. Add missing modality, source role and class variants.
- [ ] Verify patient/fold isolation and that the scorer is the only holdout-label reader.
- [ ] Run one tiny synthetic dataset through fit -> OOF -> fake Agent -> patch/replay -> fit fusion -> heldout score. All four arms share patient/input/encoder budgets.
- [ ] Perturb a relevant evidence value; measure downstream state/score change. Perturb unrelated metadata; verify stability. Removing an invalid citation cannot increase evidence eligibility.
- [ ] Distinguish fixed-base state-branch replay from full recomputation of changed audio/text. Never label the former an end-to-end causal intervention.
- [ ] Test restart/cache invariants and outputs for a failed API response. All assigned subjects remain in denominators.
- [ ] Output .local/pilot-handoff/qa/offline_acceptance.json with exact SHA, tests, missing coverage and failures.

## Acceptance
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q tests/test_pilot_integration.py tests/test_pilot_interventions.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q
```
Expected: existing and new suites pass, or documented pre-existing failures are isolated by T7. No test relaxation to hide a production bug. API invocations must equal zero.

## Required completion response
Write .local/pilot-handoff/T4/WORKER_RESULT.md with: base SHA, final SHA, changed files, exact test commands/results, artifact paths, remaining findings, schema/config changes requested, paid calls and tokens (zero for code-only tasks). Return this path and a summary under 400 words. Do not paste patient data or the entire code diff into coordinator chat.
