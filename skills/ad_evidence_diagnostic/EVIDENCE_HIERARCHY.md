# Evidence hierarchy and precedence

Apply this order during every decision check:

1. **Data validity**: the audio/transcript/segment exists and belongs to this case.
2. **Task and role applicability**: the task can measure the state and the evidence comes from the patient role.
3. **Traceability**: every important state and claim resolves to MetricEvidence and source segments.
4. **Measurement reliability**: reference support, alignment and stability are adequate; confounds are explicit.
5. **State consistency**: supporting and counterevidence are both considered; duplicate metrics do not vote twice.
6. **Ordinal evidence contract**: the Agent emits only the permitted 0-to-4 evidence scores; probability fusion remains outside the Agent.
7. **Clinical wording**: the report reflects screening evidence without biological diagnosis, unsupported stage or causal mechanism.

Lower-priority goals cannot override a higher-priority failure. For example, a plausible clinical narrative cannot rescue an invalid segment; a high supervised probability cannot make QC evidence reportable; fluent wording cannot turn an uncalibrated score into a clinical probability.

## Evidence roles

- `clinical_support`: interpretable, observable and reliable evidence permitted in support/counterevidence.
- `model_auxiliary`: may affect the supervised prior under a cap, but not the clinical rationale.
- `quality_control`: affects reliability, abstention or collection advice only.
- `planned_unavailable`: scientifically relevant but not currently measured with a validated method.
