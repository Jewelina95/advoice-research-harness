# T3: Implement Blind Agent, Evidence Trace and Bounded Replay

Model: gpt-5.6-sol; reasoning: high.
Depends on: T0.
Read START_HERE.md and CONTRACTS.md first. All repository-relative paths resolve from the assigned worktree root.
Write ownership: `src/advoice/pilot/runtime.py`, `tests/test_pilot_runtime.py`.
All source code and public UI text must be English. Do not launch unrelated work.

## Inputs
Shared contracts; existing authority_review_runtime, agent_runtime, evidence executor/tool registry, routing/observability and AD skills. Do not recreate medical skills from memory. Inventory loaded knowledge files and their hashes.

## Steps
- [ ] Adapt the existing provider and executor through an explicit new correlated-fusion route; preserve historical strict route. If a shared-file change is unavoidable, return a minimal patch to T0 rather than editing it.
- [ ] Assemble a case packet containing actual patient transcript spans, roles/tasks, source-backed measurements, states, applicable skills and confounds. Ensure modalities absent at source remain absent.
- [ ] Mask truth, supervised probability, diagnosis directory names and subject-identifying public names in prompt AND tool-visible payloads. Log sanitized prompt hashes locally.
- [ ] Separate legal cited evidence, technical reliability and predictive usefulness. Shared evidence may be cited and scored, but is not relabeled as independent new evidence.
- [ ] Parse one typed v0 assessment with class support, citations and max 3 bounded operations. Every cited ID must belong to the snapshot and be applicable to the task.
- [ ] Validate operations against permitted source-backed executor functions. Reject arbitrary metric edits, fabricated clinical facts, schema changes, duplicate operation loops.
- [ ] On accepted changes recompute the evidence -> state -> replay-score dependency closure. Capture old/new values, versions and invalidation graph.
- [ ] For meaningful changes call the same model once on v1 with no further operations. Stale v0 cannot stand in for v1; a failed second call triggers J-AS fallback but preserves auditable J-S if valid.
- [ ] Implement version-aware cache, per-request usage, timeout, bounded transport retry and global budget checks. No generation of long clinician reports.
- [ ] Export claim -> state -> evidence -> source trace plus unsupported/opaque portions explicitly.

## Tests
- [ ] Shared but legally observable evidence can influence an assessment; illegal evidence cannot.
- [ ] Increasing base uncertainty alone cannot create Agent strength.
- [ ] Invalid IDs, stale hashes, prompt injection in transcript, diagnosis-bearing paths and mismatched roles are rejected or safely isolated.
- [ ] One evidence item repeated across states is not counted as multiple independent supports.
- [ ] Agent/class direction opposite to state delta is retained for the learned fusion, not silently flipped.
- [ ] Accepted repair invalidates descendants; rejected repair does not change v0.
- [ ] Second-pass failure cannot reuse old scores; third semantic call is impossible.
- [ ] Cache misses on model, skill, reference fold, source, state or prompt changes; exact repeats are hits.
- [ ] Usage includes failed attempts; missing provider usage is not logged as zero.

## Acceptance
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q tests/test_pilot_runtime.py tests/test_agent_runtime.py tests/test_authority_review_runtime.py
```
Expected: all tests pass with a fake provider; no real patient API calls. Never assert that valid citations establish clinical correctness by themselves.

## Required completion response
Write .local/pilot-handoff/T3/WORKER_RESULT.md with: base SHA, final SHA, changed files, exact test commands/results, artifact paths, remaining findings, schema/config changes requested, paid calls and tokens (zero for code-only tasks). Return this path and a summary under 400 words. Do not paste patient data or the entire code diff into coordinator chat.
