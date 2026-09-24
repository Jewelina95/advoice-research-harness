# T0: Freeze Contracts and Integration Baseline

Model: gpt-6-astra; reasoning: high.
Depends on: none.
Read START_HERE.md and CONTRACTS.md first. All repository-relative paths resolve from the assigned worktree root.
Write ownership: `src/advoice/pilot/__init__.py`, `src/advoice/pilot/contracts.py`, `configs/pilot/evidence_state_v1.yaml`, `tests/test_pilot_contracts.py`.
All source code and public UI text must be English. Do not launch unrelated work.

## Inputs
START_HERE.md, CONTRACTS.md, current git diff; the 2026-09-23 architecture/execution plans; configs/states/audio_states.yaml and configs/metrics/audio_metrics.yaml.

## Steps
- [ ] Read current dirty changes in routing, agent_runtime, authority_review_runtime and conditional_authority. Record accepted, unresolved and unrelated changes in .local/pilot-handoff/ACCEPTED_BASELINE.md. Do not revert the user's work.
- [ ] Capture git status, SHA and binary diff locally. Audit each relevant change, stage only accepted files, create a checkpoint commit. Do not commit private report outputs or data.
- [ ] Verify existing targeted tests before adding contracts. Record failures that predate this task; do not label them newly introduced.
- [ ] Implement shared records and validation in contracts.py; reuse MetricEvidence and StateCard types. Provide explicit serializers for inference-safe vs scorer-only records.
- [ ] Implement a resolved config schema containing dataset manifests, route, model, skills, split seed, fixed hyperparameters, cache policy, call/token/cost caps. Paid run budget defaults to disabled until concrete limits and pricing are resolved.
- [ ] Write tests for labels/paths rejected at inference boundary; missing vs neutral; class order; stale snapshot; unsupported AD-stage fields; binary vs 3-class mappings.
- [ ] Freeze CONTRACTS.md and commit shared files. Create T1/T2/T3 worktrees from this exact checkpoint. Use branch names pilot/data, pilot/learning, pilot/runtime only if available; otherwise append a unique run ID.
- [ ] Integrate owned worker commits in order. A worker needing a shared-file change returns a patch proposal; only you merge it.
- [ ] After T4/T5/T6 implementation, run full offline suite and send one candidate SHA to T7. Preserve previous routes behind explicit configuration.
- [ ] Publish code/docs only after review and tests; never push .local, raw data, source transcripts, API credentials or copyrighted example audio without distribution rights.

## Acceptance
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q tests/test_pilot_contracts.py tests/test_agent_runtime.py tests/test_authority_review_runtime.py tests/test_conditional_authority.py tests/test_conditional_contracts.py
git diff --check
```
Expected: all listed tests pass; no paid calls; no silent changes to default historical behavior. A dirty starting tree is not permission to include all files in the checkpoint.

## Required completion response
Write .local/pilot-handoff/T0/WORKER_RESULT.md with: base SHA, final SHA, changed files, exact test commands/results, artifact paths, remaining findings, schema/config changes requested, paid calls and tokens (zero for code-only tasks). Return this path and a summary under 400 words. Do not paste patient data or the entire code diff into coordinator chat.
