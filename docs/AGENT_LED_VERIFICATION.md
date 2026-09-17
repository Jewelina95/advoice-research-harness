# Agent-led implementation verification

## Scope

This change introduces a separate Agent-led inference path and does not replace
historical experiment outputs or claim a better clinical benchmark. No clinical
cohort training or held-out performance evaluation was performed in this change.

The original bounded-correction path is retained as an explicit comparator.
The new path starts from an existing evidence workspace, not from a newly
implemented waveform extractor. Missing measurement replay is not simulated.

The supervised modules are upstream evidence-organization components, not two
mandatory votes that the Agent must follow. They may learn task-conditioned
metric-to-state relations, reliability and state relevance when suitable labels
are available. Their frozen class probabilities remain optional, correlated
context. The Agent's required input is the MetricEvidence/StateCard/segment
evidence graph; its required output is an evidence-linked decision or abstention
with an auditable trace.

## Software checks

On 2026-09-17, the full Python test suite passed: **453 tests**. Dependencies emitted
existing pandas/scipy/pkg_resources/OpenBLAS warnings; these were not hidden.
`git diff --check` passed.

An independent Agent review found six issues, addressed with regression tests:

1. Imported reviewed evidence must not reactivate stale supervised outputs.
2. A filesystem-capable Codex provider cannot satisfy the blinded tool boundary;
   it is disabled on the new path.
3. State-local counterevidence must be included in inspection and validation.
4. Parent report permission does not authorize prohibited child measurements.
5. Provider and frozen-advisor identities must participate in calibration identity.
6. Abstention cannot cite uninspected findings or publish them as clinical findings.
7. The Agent-led entry point must load the complete existing AD knowledge package,
   not only its execution overlay.
8. A supplied patient transcript is an inspectable evidence context and must be
   inspected before finalization; it never replaces typed evidence citations.
9. Legacy untyped evidence IDs are canonicalized before tool access.
10. State inspection marks the returned child metrics and segments as inspected;
    hypothesis validation checks both support and counterevidence immediately.
11. A provider response containing several concatenated actions executes only the
    first action, preserves the raw response and records discarded future actions.

Other checks cover final decisions that disagree with supervised advisors,
source-hash sensitivity to changed values, dependent-state invalidation,
reinspection after revision, request failure without fallback, exact evaluation
cohorts, abstention denominators, and calibration overlap rejection.

## Real provider check

The public synthetic fixture was sent to `gpt-5.6-luna` through `openai_api`.
The first live check executed seven valid actions: quality inspection, two state
inspections, metric inspection, hypothesis recording, counterevidence inspection,
and abstention. It abstained because the artificial evidence did not justify
distinguishing the research classes. No patient recording was transmitted.
A second live check after the review fixes also completed seven valid actions
and abstained, with no supervised fallback or rejected tool action.

This demonstrates actual provider/tool execution, not diagnostic accuracy or
superiority over SpeechCARE. Synthetic labels must not be used to estimate
clinical performance, and abstention must not be counted as a correct class.

An authorized single-case IAEAV integration check then exercised the complete
knowledge package, a 469-token Spanish transcript and forced-choice benchmark
mode. It inspected quality, six state views, counterevidence and the transcript,
then produced the correct held-out `AD` class in 11 provider requests. This is an
interface/integration result only. A selected single case is not an estimate of
accuracy, generalization, clinical validity or superiority over SpeechCARE.

## Remaining validation

Freeze model, skill, tools, advisor artifacts, cohort and test protocol before a
matched comparison. Validate coverage as well as accuracy and ranking metrics.
Measure robustness to ASR, role and task errors and adversarial transcript text;
schema validation alone does not establish semantic grounding or medical safety.
Probability calibration needs separate development data. More capable future
models still require revalidation rather than an assumed performance increase.
