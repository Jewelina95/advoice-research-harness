# Agent cost audit and calibration decision

Scope: inspect saved artifacts, repair call accounting, and specify calibration.
No new patient inference, training, or full-cohort evaluation was run. Clinician
report generation remains a future formal output stage; it stays disabled in
the current decision evaluation and demo. It is not being removed from the
product. Decisions must be frozen before narrative generation.

## Previously completed results

These are selected, previously inspected development pilots, not untouched
test evidence or performance of the latest formula repair.

| Dataset | Cases | Frozen accuracy | Earlier fused accuracy | Important limitation |
| --- | ---: | ---: | ---: | --- |
| PREPARE | 9 | 7/9, 77.8% | 8/9, 88.9% | Seven HC, two AD, no MCI; micro AUROC 0.8889 to 0.8765; ECE 0.2384 to 0.3910 |
| NCMMSC long track | 3 | 1/3, 33.3% | 1/3, 33.3% | MCI and AD remain interchanged; multiclass Brier 0.7232 to 0.7329 |
| ADReSSo diagnosis | 3 | 2/3, 66.7% | 2/3, 66.7% | Brier 0.4455 to 0.1827; improved probabilities, unchanged accuracy |

Single-case smoke checks: ADReSS 2020 1/1 to 1/1; PROCESS-2 0/1 to 1/1;
IAEAV 0/1 to 0/1; public figures 1/1 to 1/1. These establish neither stable
channel performance nor generalization. None establishes superiority to
SpeechCARE under its full evaluation protocol.

Source aggregates, relative to the local worktree:

- `.local/authority_expanded_pilot_9/PREPARE_DrivenData/aggregate.json`
- `.local/authority_expanded_pilot_v3/NCMMSC2021_AD/aggregate.json`
- `.local/authority_expanded_pilot/ADReSSo_2021_diagnosis/aggregate.json`
- `.local/authority_channel_routed_smoke/*/aggregate.json`

The later label-free formula replay changes one saved class in each of the
PREPARE and NCMMSC pilots. It does not establish that either change is correct.

## Measured cost contributors

Counts below come from local files, not a provider billing export. Bytes are
not tokens; saved outputs are not a count of all attempted/billed requests.

1. Legacy tool-loop outputs in `conditional_authority_pilots` contain 44 saved
   Luna responses for three cases (15, 14, 15) and 32 Sol responses (10, 12, 10).
   These were multiple action rounds per case, not three single-call decisions.
   They are a different runner from the current blind/advisor two-call runtime.
2. Seven current authority-cache directories contain 69 saved outputs with 69
   distinct filenames. This audit found no duplicate output filenames across
   these directories; it cannot claim exact duplicate paid requests occurred.
   Cache lookup is nevertheless local to an output path, so moving a request
   into another cache directory does not reuse an existing identical result.
3. `_policy_documents` reads all 11 Markdown files in the AD evidence skill.
   Their contents total 24,193 characters; the serialized UTF-8 document map
   is 24,903 bytes. Both blind assessment and advisor reconciliation resend it.
   The package includes report instructions and references during decision-only
   testing. Medical scope and uncertainty rules still belong in decision calls;
   report prose instructions need not.
4. Saved JSON schemas range from 2,112 to 82,047 bytes, median 26,483. In the
   largest PROCESS-2 advisor schema, 92 distinct MetricEvidence IDs appear 368
   times across action alternatives. The repeated IDs alone occupy 64,600
   characters. This is exact structural duplication, not additional evidence.
5. The old OpenAI path constructed `OpenAI()` with SDK default `max_retries=2`
   inside a five-attempt application retry loop. For retriable failures this
   permits up to 15 HTTP attempts. It does not mean every attempt was billed.
   The installed SDK also defaults to a 600-second read timeout. The Codex
   wrapper has a 1,200-second subprocess timeout.
6. The OpenAI output ceiling is 12,000 tokens; it is a ceiling, not measured
   generation. The Codex path has no explicit reasoning/output budget override.
   Both paths previously discarded provider token accounting. The Codex path
   also discarded successful stdout, preventing an audit of internal events.

No evidence here proves that schema size, retries, or model reasoning alone
accounts for a particular share of the user's total account consumption.
Engineering conversations and reviewer agents consume tokens separately from
patient inference. Historical billing cannot be reconstructed precisely from
these response JSONs.

## Changes implemented in this audit

- Each provider request now appends a `.calls.jsonl` accounting sidecar with
  provider/model, attempt, request hash, input/schema sizes, elapsed time, and
  numeric token usage when returned. No transcript, prompt, API key, or error
  message is copied into this ledger.
- Codex uses JSONL events; only `turn.completed` numeric usage is retained,
  including cached input tokens when supplied. Missing events are marked
  unavailable rather than treated as zero.
- OpenAI saves input/output/cached/reasoning token counts when supplied. Failed
  calls have unknown usage, not zero. SDK retries are disabled; the application
  owns the single bounded retry policy.
- Exact-path cache hits record zero provider calls without overwriting the
  original request usage. Usage entries describe successful returned responses,
  not a guaranteed complete billing ledger for failed network requests.

The transport contract is now compacted without weakening the server-side
validation. Full MetricEvidence IDs remain in the evidence registry and audit,
while the provider sees stable case-local IDs such as `E001`. Provider outputs
are translated back before validation. Action schemas now define each state's
allowed evidence once; action-to-multiplier compatibility remains enforced by
the typed local parser. On 71 comparable saved schemas, serialized size falls
from 2,017,521 to 339,105 bytes in aggregate (83.2%). The largest PROCESS-2
advisor schema falls from 82,047 to 5,745 bytes (93.0%). These are measured
schema bytes, not a promise of the same percentage reduction in billed tokens.

Decision-only policy loading excludes the bibliography and report prose
contract while retaining medical scope, state knowledge, task observability,
confounds, evidence hierarchy, rollback, leakage and report-permission rules.
The provider output ceiling is reduced from 12,000 to 4,096 tokens. The bounded
application retry schedule is reduced from five attempts to three (0, 20 and
60 seconds), with SDK retries disabled, so one logical OpenAI decision call can
make at most three transport attempts rather than fifteen.

The patches do not change evidence values, predictive weights, model choice,
or clinical policies. Mock-provider tests verify accounting without paid calls.
Verification: the full local suite passed with 770 tests and 3 skips;
`git diff --check` passed. Existing pandas, scikit-learn and numerical-library
compatibility/deprecation warnings remain and were not treated as failures.

## Cost reductions to validate before live evaluation

Preserve all patient evidence and traceability. Compact only redundant transport
representation: give evidence stable case-local short IDs with a reversible
server-side map; reference state-bound ID definitions once in schemas; select
decision-relevant skill sections without deleting medical limitations. Keep the
runtime identity/hash tied to the exact compacted payload, schema, model and
policy. Test invalid, cross-state, stale, and unknown citations after compaction.

Do not replace blind/advisor passes with a probability-anchored single prompt
just to save tokens. Keep a hard decision-call budget, shared content-addressed
cache, configurable timeout/output budget, and bounded retries. A report call
must not run during these evaluations. Only after offline contract tests pass,
measure a small frozen cohort's real usage and fidelity before scaling up.

For engineering work, delegate bounded file-level tasks with short manifests,
not the entire historical conversation. Reuse completed audits and run targeted
tests before broad suites. No additional reviewer windows were opened for this
cost audit.

## Weight design: calibrated conditional pooling

The objective is complementary correctness, not maximizing Agent influence.
Supervised uncertainty does not establish Agent reliability. A contradictory
Agent may be correct, so disagreement alone must not zero its independent path.
An invalid citation or impossible state revision is an admissibility failure,
which is different from a disagreement between valid predictions.

Recommended next candidate, not implemented or fitted by this audit:

`p_final = (1 - g(x)) * p_state + g(x) * p_agent`

- `p_state` is the frozen predictor recomputed once after validated state
  repairs; it equals the original prediction when no repair is admissible.
- `p_agent` comes from the blind assessment of transcript, MetricEvidence,
  StateCards and AD skills. Current ordinal scores are not probabilities or
  likelihood ratios. Fit a low-capacity regularized score calibrator on
  development outcomes before using them as a distribution.
- `g(x)` is a small regularized gate, not a constant selected by preference.
  Features include observable-state coverage, verifiable citation validity,
  transcript/role quality, disagreement, and task/language indicators with
  shrinkage toward a shared gate. Confidence or margin is not a substitute for
  these features. Invalid Agent evidence makes its branch ineligible.
- Fit the two correlated branches jointly under a proper scoring objective
  such as log loss, with Brier, class recall, and accuracy as predeclared checks.
  Do not multiply their probabilities as if they were independent evidence.
  The repaired state model already includes Agent edits; do not add the same
  edits a second time as a separate likelihood factor.

Begin with a constant convex pooling baseline and enable a small conditional
gate only if subject-disjoint inner validation supports it. Nine inspected
PREPARE examples and three Chinese examples cannot fit or select a dependable
task/language gate. Collect or reuse frozen development predictions, not a new
full encoder training run, for this calibration exercise.

Keep HC-versus-impairment and MCI-versus-AD distinct. The current formula freezes
MCI:AD odds, so weight tuning cannot repair MCI/AD reversals. A three-class
extension needs a declared endpoint, adequate stage-labeled development cases,
and class-wise evaluation; do not claim it exists already.

Implementation order: freeze participant splits and model/skill versions;
produce out-of-fold development predictions from the base model; calibrate
branch scores and select gate capacity inside the development folds; freeze
the selected artifact; then run the untouched protocol-matched test once.
Previously inspected test examples must be disclosed as development-exposed.
Always compare predictor-only, Agent-only, replay-only and pooled decisions on
the same patients. Count both corrected and newly harmed cases. Report patient
bootstrap uncertainty, full confusion matrices, micro/macro AUROC, log loss,
Brier and accuracy rather than selecting only the metric that improved.

This design preserves meaningful Agent authority, including potentially high
weights when validated, without promising that a more powerful LLM necessarily
improves a given clinical dataset.

## Related methods and limits of borrowing

- SpeechCARE learns adaptive fusion over modality representations. Our proposed
  gate instead combines a traceable state prediction with a blind Agent
  assessment; its weights must be evaluated at the decision level. Neither
  finer granularity nor the presence of an Agent implies superiority.
  https://www.nature.com/articles/s41746-025-02026-x
- Guo et al., ICML 2017: temperature scaling calibrates probabilities. A common
  positive scalar temperature preserves the argmax, so calibration alone does
  not fix misclassified cases. Pooling/routing can change the decision.
  https://proceedings.mlr.press/v70/guo17a.html
- Mozannar and Sontag, ICML 2020: learn a predictor and deferral policy from
  expert decisions with a cost-sensitive objective. The relevant idea is
  learning when another decision-maker helps, not trusting it whenever the
  base model is uncertain. Our soft pooling proposal is not their exact
  algorithm and inherits no automatic clinical guarantee.
  https://proceedings.mlr.press/v119/mozannar20b.html
