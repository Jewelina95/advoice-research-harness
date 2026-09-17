# Agent-led evidence judgment

## Change in decision authority

The new `advoice agent-led` path makes the Agent responsible for the final
research classification and for choosing evidence-inspection actions. It is not
the old prior plus a scalar correction. The existing `run`/`experiment` paths
remain the explicitly separate bounded-correction comparator. Historical scores
do not describe this new runtime. No accuracy gain or clinical validation is
claimed by this implementation.

```
audio / transcript / task / language
  -> existing preprocessing, metrics, state cards and segment provenance
  -> versioned evidence workspace
  -> Agent: inspect -> hypothesize -> optionally consult learned models
             -> revise evidence -> inspect again -> decide / abstain
  -> independent probability calibration (when available)
  -> research report + executed tool trace + separate outcome evaluation
```

## Why this is a different fusion method

The old decision has the form `p_final = fusion(p_supervised, agent_scores, gate)`.
Even valid evidence revisions cannot change it unless a numerical correction
passes every gate. Its output therefore remains structurally anchored to a
supervised prior.

The new policy is `a_t = Agent(E_t, tool_history)` and `E_(t+1) = Tool(E_t, a_t)`.
The final class and ordinal evidence scores come from the Agent's validated
decision on the current snapshot, not from automatic interpolation with a prior.
The two trained modules are optional advisors, requested only after an
evidence-grounded initial hypothesis. They are correlated estimates based on
overlapping inputs and must not be counted as independent clinical evidence.

This is sequential evidence integration, not a new claim that an LLM computes a
valid Bayesian posterior. Additional reasoning cannot recover information that
was discarded upstream. Any predictive advantage must arise from better use of
available evidence, identification of measurement errors, or useful additional
tool results. More steps or a larger model can also introduce errors.

## What happens to the two trained modules

The existing trained module A (branch fusion) and module B (bounded supervised
residual) remain available as snapshot-bound advice. Their saved outputs are
exposed separately through `consult_models`; no training occurs in this command.
This release does NOT relabel them as clinically supervised state estimators,
nor does it pretend to have new clinician labels. Rule-based state aggregation
and learned class associations remain distinct.

An imported prediction is usable only with `advisor_provenance.evidence_hash`
matching `hash_values([evidence_snapshot(workspace)])` and nonempty artifact
identities in `advisor_provenance.artifacts.module_a` / `module_b`. The producer
must bind these immediately after inference on that exact evidence, using the
actual frozen artifact hashes. Merely adding a new hash to old predictions is
not recomputation. Older unbound workspaces remain usable for Agent-only
judgment, but their numeric advisors are unavailable. This is a producer contract,
not independent verification that an external artifact was trained correctly.

After evidence changes, cached A/B outputs are unavailable. They are not
automatically recomputed, and the Agent cannot cite them as current estimates.
The Agent can inspect the revised evidence and form a new decision without those
advisors. Full raw-input replay for A/B remains separate work; do not bypass the
stale-output boundary to simulate it.

## Executed cognition mechanisms

1. **Independent initial hypothesis:** the initial input is an allowlisted,
   recursively sanitized evidence index. Supervised predictions are inaccessible
   until an evidence-grounded hypothesis has been recorded.
2. **Question-directed inspection:** the Agent executes real state, metric,
   segment, quality, task-comparison and counterevidence tools. A precomputed
   plan does not count as execution. Segment tools currently return stored
   transcript/timing metadata, not audio-model listening.
3. **Revision and renewed judgment:** downweight/invalidate/unavailable actions
   create immutable snapshots, withdraw affected citations, invalidate model
   advice and require a new inspection/hypothesis. Scores are not manually
   rewritten by the Agent. The final decision can disagree with both advisors.
   If a withdrawn metric also supports another state, that dependent aggregate
   is withdrawn until recomputation rather than retaining its stale score.

These mechanisms adapt the project's cognition-inspired state-maintenance idea.
They are not an exact reproduction of Cognition-of-Thought or proof of its
social-reasoning results transferring to clinical screening.

## Quality, decision and release are different

The quality tool reports measured confidence/missingness, whether source objects
exist, and potential confounds. It does not turn unknown medical history into
absence or assign an invented clinical-reliability score. No universal
`unknown confound => zero Agent authority` multiplier is used on this research
path. Unknowns remain visible limitations. The Agent must inspect quality and
counterevidence before deciding.

Deterministic validators reject unknown IDs, quality-as-disease citations,
uninspected evidence, stale revisions and inconsistent scores. This establishes
structural validity only; it does not prove that a cited finding supports the
medical interpretation. All outputs retain `clinical_release=false`.

Provider errors, exhausted budgets, and abstention produce no substitute class.
The prior is never silently reported as an Agent decision. Ordinal scores are
not risk probabilities. Probability calibration must use a separate development
cohort and be bound to the exact Agent/skill/policy version; upgrading the model
invalidates that calibration. Test diagnoses never enter requests or tools.
The model fingerprint includes the skill, label order, tool budget, and decision,
validation, revision and provider source code contents; an inference-policy
change therefore invalidates an earlier calibration identity too.
Provider choice and declared frozen-advisor artifact identities are also bound.

## Stable interfaces, replaceable model

Keep evidence IDs, state/task definitions, permission rules, tool contracts and
evaluation cohorts versioned. Replace the Agent model through configuration,
then verify accuracy, abstention coverage, grounding and tool failures. A more
capable model may choose better evidence and identify contradictions, but an
automatic monotonic clinical improvement over one to three years is not assumed.

There is no GPT fine-tuning or post-training in this release. Existing supervised
models supply learned estimates; the Agent performs inference-time tool use.
Model upgrades are not training, and prompt changes are not post-training.

## Run without changing existing experiments

```bash
advoice agent-led \
  --workspaces /private/artifacts/DATASET/diagnostic_agent_workspaces.jsonl \
  --labels HC MCI AD \
  --provider openai_api \
  --model MODEL_AVAILABLE_TO_YOUR_API_ACCOUNT \
  --max-cases 1 --max-steps 16 \
  --output-dir /private/runs/agent-led-pilot
```

`OPENAI_API_KEY` is required. External inference sends the sanitized evidence
objects to the configured provider; do not use participant data without the
necessary permission. Local paths and diagnosis labels are removed, but speech
transcripts can still contain sensitive content. The runtime does not claim
automatic de-identification of free text. The disabled provider produces an
explicit unavailable result, not a fake model answer. The output directory must
be new, preserving historical artifacts.
`codex_cli` is not available for this blinded decision path: a tool-capable CLI
with filesystem access could read labels or priors outside the audited tools.

A public synthetic fixture is at `demo/agent_led/synthetic_workspace.jsonl`.
It is software-test input, not patient data. For held-out evaluation, `--truth`
accepts a JSON mapping of exact case IDs to labels, read only after inference.
The selected cohort and truth map must match exactly; an optional case limit
is deterministic and independent of diagnosis.

Outputs include `decisions.jsonl`, `run.json`, `action_schema.json`, the exact
skill, per-call responses, `report.html`, and `evaluation.json` when labels are
provided. Layer A evaluates prediction; Layer B records execution/grounding
checks, not clinician-rated report quality. Abstentions stay in the all-case
denominator. Ranking metrics on ordinal scores are not probability calibration.

## Remaining boundaries before a performance claim

- Add measurement adapters for speaker/ASR/alignment corrections with validated
  recomputation; current tools inspect existing measurements and revise states.
- Attach full frozen supervised replay if numerical advice is required after a
  state revision. Current stale advisors deliberately remain unavailable.
- Validate Agent interpretations and version-specific probability calibration.
- Compare matched encoders, participants, tasks and label mappings on untouched
  data; the repeatedly inspected PREPARE test remains retrospective.
- Do not present software tests or synthetic live calls as better-than-SpeechCARE
  results. The expected benefit is a hypothesis to test, not an implementation fact.
