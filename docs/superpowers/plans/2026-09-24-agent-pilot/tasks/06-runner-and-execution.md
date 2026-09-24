# T6: Implement Resumable Runner, then Execute Authorized Pilot

Model: gpt-5.6-terra; reasoning: high.
Dispatch: T6A depends on T4/T5; T6B depends on T7P ACCEPT. Execute only the assigned phase.
Read START_HERE.md and CONTRACTS.md first. All repository-relative paths resolve from the assigned worktree root.
T6A source ownership: `src/advoice/pilot/runner.py`, `scripts/run_evidence_state_pilot.py`, `tests/test_pilot_runner.py`. T6B owns local run artifacts only and cannot modify tracked source.
All source code and public UI text must be English. Do not launch unrelated work.

## Inputs
Integrated artifacts and tests, immutable manifests, contracts, configured provider and pricing. Read T7 review before any paid run. This task has implementation and execution phases; stop between them for independent review.

## Phase A: implement offline
- [ ] Implement stages inventory, split, canary, fit-oof, assess-oof, fit-fusion, fit-final, predict-holdout, stress, model-comparison, score, render; stage all follows their DAG. Canary is a required gate before analytical model calls, uses external engineering cases and never changes frozen analytical membership.
- [ ] Implement --config, --run-dir, --stage, --resume, --dry-run, --provider fake|configured, --allow-paid. Fake is testing only and prominently stamped; real clinical result requires configured provider.
- [ ] Every stage writes input/output hashes, code SHA, version, start/end, row counts and status atomically. Interrupted stages resume without losing denominators or duplicating successful calls.
- [ ] Use T1 manifests, T2 fits/calibrators and T3 real provider adapter. Prevent old cached predictions from masquerading as freshly fitted OOF.
- [ ] The scorer joins truth only after prediction hashes are frozen. Compute paired subject-level bootstrap CIs (2000 seed-fixed replicates), per-class metrics, calibration and help/harm. Undefined AUROC has a reason, not 0.
- [ ] Enforce capability manifests: development labels only for split/training/calibration; holdout/stress labels only for score. assess-oof and predict-holdout have no label reader. Validate provider payloads and same-fold reference_fit_id at runtime, not only in tests.
- [ ] Freeze OOF row hashes and calibrators before full-development refit. Reject final-fit IDs as calibration training inputs; fit-final exists only to prepare heldout inference.
- [ ] For accuracy report exact paired discordance counts and McNemar test; small pilot findings remain exploratory. Model x framework comparison exploratory on 24 development subjects, not a heldout causal claim.
- [ ] Validate call/token/USD budgets, actual available model ID and gate evidence before --allow-paid. Resolve credentials without printing them.
- [ ] Produce tests for no-paid-by-default, restart idempotency, hash mismatch, budget stop, deadline stop and no holdout-label access before score.
- [ ] Submit candidate SHA to T7. Do not run canary until ACCEPT.

## Phase B: real execution after T7 preflight ACCEPT
- [ ] Resolve a capable available clinical inference model; nominal target Sol, but record the actual provider-supported ID. Do not assume programming-agent names are API IDs. No silent replacement when unavailable.
- [ ] Run 24 cached/synthetic engineering cases first, then up to 12 external-to-pilot engineering canaries across four channels, max 24 semantic calls. If a channel has no eligible engineering case, use fixture and record real-canary gap; do not consume holdout for tuning.
- [ ] Resolve real limits before calls: total semantic cap 910; retries maximum one per transport request; initial concurrency 2; input ceiling 8000 tokens, output+reasoning ceiling 4096 per request if supported. Over-context goes to a deterministic source-preserving packet compactor; never silently truncate the cited span.
- [ ] Default spending stop: USD 10 canary; USD 100 combined subsequent pilot/model comparison. A cap is a stop, not permission to fake completion. If pricing/usage cannot be monitored, stop paid queue and return the missing telemetry.
- [ ] Compute remaining ETA from measured provider p50/p95 and effective concurrency plus local fit timings. Log progress per completed subject, no repeated narrative from another Agent.
- [ ] Run PREPARE OOF -> decisions -> calibration -> final fit -> frozen 30 holdout; run ADReSS and NCMMSC independently in the same order. Do not stop/reselect on an ordinary low score.
- [ ] Run 14 IAEAV + 6 DementiaNet stress cases without cross-dataset risk-calibration claims.
- [ ] Run 24-patient model comparison: exact same patient inputs/skills/budgets under two real model IDs x flat/structured representation. Structured path max2 calls, flat1; cap144. Use development only and report differential compute cost.
- [ ] Score and render locked artifacts. Do not start the remaining databases or the official PREPARE benchmark in this task.
- [ ] Return observed results, uncertainty, fallback/failed counts, usage, ETA for next formal study and paths. A negative result is complete evidence, not a reason for unbounded iteration.

## Proposed CLI, runnable only AFTER Phase A exists
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q tests/test_pilot_runner.py
PYTHONPATH=src python scripts/run_evidence_state_pilot.py --config configs/pilot/evidence_state_v1.yaml --run-dir .local/pilot-v1 --stage all --provider fake --dry-run
PYTHONPATH=src python scripts/run_evidence_state_pilot.py --config .local/pilot-v1/protocol/resolved_config.yaml --run-dir .local/pilot-v1 --stage all --provider configured --allow-paid --resume
```
Dry-run is structural planning only, not data inference. Paid command must validate T7 gate tied to current SHA/config before execution.

## Required completion response
This task file is dispatched twice: T6A implements Phase A only; T6B executes Phase B only after T7P ACCEPT. Write .local/pilot-handoff/<T6A-or-T6B>/WORKER_RESULT.md with: base SHA, final SHA, changed files, exact test commands/results, artifact paths, remaining findings, schema/config changes requested, paid calls and tokens (zero for T6A). T6B cannot change tracked code. Return the path and a summary under 400 words, without patient data.
