# Report Permission Policy

Version: `report-permission-1.0`

`report_permission` is computed before the diagnostic Agent runs. The Agent may
inspect blocked evidence for quality review or rollback, but it may not promote
that evidence into a clinical finding.

## Deterministic conversion

Each MetricEvidence carries the registry-level permission and one computed
`report_permission_basis`.

1. Registry value `no` always becomes `report_permission=false` with
   `blocked_registry_no`.
2. Registry value `yes` becomes `true` with `registry_yes` only when the metric
   is observable, non-missing, clinical-support evidence, linked to the same
   task/session segments, supported by a valid fold-internal reference and at
   or above the case reliability threshold. Otherwise it becomes `false` with
   the applicable blocking basis.
3. Registry value `conditional` follows the same checks and additionally
   requires a registered task-, language- and method-specific implementation
   that passed its metric validation tests. The evidence must cite a
   `metric_validation_id` resolving to a matching validated record in the case
   package; self-declaring `conditional_validated` is insufficient. Passing evidence is marked
   `conditional_validated`; otherwise it is blocked.
4. `quality_control`, `model_auxiliary` and `planned_unavailable` evidence is
   never reportable, regardless of its predictive association.
5. Evidence with a blocking confound, an unavailable observation, a missing
   value, inadequate reliability or an unvalidated planned implementation is
   never reportable.

Allowed basis values are:

- `registry_yes`
- `conditional_validated`
- `blocked_registry_no`
- `blocked_condition`
- `blocked_role`
- `blocked_missing`
- `blocked_reliability`
- `blocked_unavailable`
- `blocked_planned`

The case-package builder owns this conversion. The validator recomputes the
hard parts of the rule and rejects contradictions. The diagnostic Agent cannot
change `report_permission` or `report_permission_basis`.
