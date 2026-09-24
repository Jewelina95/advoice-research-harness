# T7: Independent Preflight and Results Review

Model: gpt-6-astra; reasoning: high.
Dispatch: T7P depends on T6A; T7R depends on T6B. Execute only the assigned review phase.
Read START_HERE.md and CONTRACTS.md first. All repository-relative paths resolve from the assigned worktree root.
Write ownership: review artifacts under .local/pilot-handoff/review only; no tracked source edits.
All source code and public UI text must be English. Do not launch unrelated work.

## Scope
Read-only independent reviewer. Do not implement fixes, rerun APIs or redesign the experiment. Review the candidate commit and actual artifacts, not worker assertions.

## Preflight review, before T6 Phase B
- [ ] Verify T0 accepted checkpoint includes needed dirty fixes and no unrelated/private files.
- [ ] Inspect diff and call graph from runner to actual predictor/provider; confirm no fake/frozen-historical substitution in real mode.
- [ ] Verify official/holdout boundary, fold-local references, base vs state replay distinction, class mapping and bound calibration inputs.
- [ ] Verify Agent sees source evidence and skills, cannot see truth/base predictions, and can genuinely affect output through learned scores or validated replay.
- [ ] Inspect shared-evidence legality vs independence, uncertainty-vs-evidence distinction, opposite-sign combinations, stale score invalidation and fallback denominators.
- [ ] Confirm matched B calibration and matched-fit sensitivity, not just higher J-AS accuracy from extra calibration.
- [ ] Verify request/retry/model/cost caps and no unbounded Agent repair loop.
- [ ] Check frozen limits: S09/S13 gaps and opaque encodings are not promoted to validated clinical constructs.
- [ ] Write .local/pilot-handoff/review/preflight.json: verdict ACCEPT|REVISE|BLOCK, code_sha, config_hash, tests, findings with paths/lines, exact return owner T1..T6. Unresolved P0/P1 => no paid execution.

## Results review, after T6 execution
- [ ] Count every assigned subject, failed request and fallback; validate no resampling after outcomes.
- [ ] Recompute a sample of predictions/metrics from saved fitted artifacts and hashes.
- [ ] Separate trained-pilot and stress outcomes; preserve negative results.
- [ ] Determine whether Agent gain is measured, uncertain or unsupported. Do not infer model-upgrade benefit merely from model names or a positive point estimate.
- [ ] Audit traceback interventions and distinguish source provenance from prediction faithfulness.
- [ ] Confirm no “better than SpeechCARE” claim from internal pilot or unmatched historical runs.
- [ ] Write final_review.json and a concise prioritized issue list; recommend targeted next work, not automatic nine-dataset retraining.

## Acceptance
Review files reference actual SHA, config and artifacts. Findings are actionable and independently evidenced. “All tests pass” alone is not scientific approval. This is a bounded review; read the task package and changed symbols, not the full user conversation.

## Required completion response
This task file is dispatched twice: T7P preflight only, then T7R results only. Write .local/pilot-handoff/<T7P-or-T7R>/WORKER_RESULT.md plus the named preflight/final_review artifact. Include candidate SHA/config hash, reviewed tests/artifacts, findings and return owner. No tracked edits or paid calls. Return the path and a summary under 400 words.
