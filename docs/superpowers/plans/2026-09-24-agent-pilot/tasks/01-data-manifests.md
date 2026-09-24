# T1: Audit Data and Freeze Subject Manifests

Model: gpt-5.6-terra; reasoning: high.
Depends on: T0.
Read START_HERE.md and CONTRACTS.md first. All repository-relative paths resolve from the assigned worktree root.
Write ownership: `src/advoice/pilot/data.py`, `tests/test_pilot_data.py`.
All source code and public UI text must be English. Do not launch unrelated work.

## Inputs
Shared contracts; configs/datasets/{PREPARE_DrivenData,ADReSS_2020,NCMMSC2021_AD,IAEAV,DementiaNet_PublicFigures}.yaml; historical 8.27/artifacts; the raw-data root. Read existing dataset adapters through the code graph. Do not recursively decode every audio file.

## Steps
- [ ] Build source_inventory with actual paths, versions, audio/transcript availability, label source, task mapping, subject count and source_group IDs. PREPARE is not a top-level raw folder: resolve its actual adapter path; never guess it from the display name.
- [ ] Verify historical counts in START_HERE against current artifacts. Match the PREPARE invalid-record exclusions by UID, not merely a total of 33.
- [ ] Inspect patient grouping across recordings, dataset aliases and duplicate hashes. NCMMSC 395 recordings are not 395 independent patients. Exclude the entire six-second track by source track metadata/path, not by indiscriminately deleting every short segment.
- [ ] Separate immutable source manifest from provider-safe pseudonymous manifest and labels.csv. Local raw paths remain only in the protected source index.
- [ ] Select PREPARE 150 -> 120/30, ADReSS 81 -> 65/16, NCMMSC 120 -> 96/24 with seed 20260923. Use class/language stratification where feasible, group identity first, deterministic documented fallback for rare strata.
- [ ] Keep IAEAV 14 and DementiaNet 6 as stress cohorts with existing acquisition/person boundary. Do not synthesize a random 80/20 split for these.
- [ ] Freeze five grouped OOF folds for each development cohort. Training and heldout membership are disjoint at subject, duplicate-group and source identity level.
- [ ] Qualify caches: raw deterministic metrics and unadapted encoder outputs may be reusable; globally fitted references, old state normalization or in-sample predictions are not.
- [ ] Output exclusions.csv, identity_audit.json, cache_provenance.json, split_manifest.csv, folds.csv and sha256 manifest under .local/pilot-handoff/data/. Unresolved identity collisions block the affected cohort.
- [ ] Canary cases must be outside the analytical 351+20 cohort when used to tune code/prompts. Select from unused historical development sources or engineering fixtures, never reserved official test.
- [ ] Reserve analytical membership first: development=281 (120+65+96), holdout=70 (30+16+24), stress=20 (14+6). Then choose up to12 engineering_canary patients outside those sets and reserved tests. If insufficient, reduce the real canary count and state the gap; never borrow or replace an analytical subject. Assert all four sets are disjoint and report grouping-adjusted counts before calls.

## Tests
- [ ] Duplicate recording on opposite partitions is rejected.
- [ ] Label-bearing filename never appears in serialized provider payload.
- [ ] Same seed/inventory yields identical manifest bytes.
- [ ] Six-second source track is excluded; long track preserved; original files never deleted.
- [ ] Historical test exclusion, source cohort mismatch and missing raw/transcript provenance are explicit.
- [ ] Missing class in an OOF fit partition fails before training, not after API calls.

## Acceptance
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q tests/test_pilot_data.py tests/test_pilot_contracts.py
```
Expected: synthetic fixtures pass; actual manifest audit gives exact or predeclared grouping-adjusted counts. No model API, encoder training, fusion modification or patient-level result upload.

## Required completion response
Write .local/pilot-handoff/T1/WORKER_RESULT.md with: base SHA, final SHA, changed files, exact test commands/results, artifact paths, remaining findings, schema/config changes requested, paid calls and tokens (zero for code-only tasks). Return this path and a summary under 400 words. Do not paste patient data or the entire code diff into coordinator chat.
