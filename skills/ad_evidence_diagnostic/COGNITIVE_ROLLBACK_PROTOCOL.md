# Cognitive rollback protocol

This protocol operationalizes a Cognition-of-Thought-inspired monitor over clinical evidence decisions.

## Violation codes

- `V01_LABEL_LEAKAGE`: hidden label, diagnostic filename or source diagnosis used.
- `V02_QC_AS_DISEASE`: quality/model-only evidence used as disease support.
- `V03_UNOBSERVABLE_STATE`: task-inapplicable state used.
- `V04_INVALID_REFERENCE`: cited evidence/segment is absent or belongs to another case.
- `V05_PERMISSION_VIOLATION`: non-reportable evidence cited clinically.
- `V06_COUNTEREVIDENCE_OMITTED`: material available counterevidence not inspected.
- `V07_UNSUPPORTED_DIAGNOSIS_OR_STAGE`: biological diagnosis/stage asserted.
- `V08_CLASS_SCORE_MISMATCH`: evidence class does not match the highest ordinal evidence score.
- `V09_REPEATED_FAILURE`: the same violation recurs after one rollback.

## Procedure

1. Record the candidate step and all cited IDs.
2. Run precedence checks in fixed order.
3. On violation, identify the earliest invalid evidence/state/hypothesis step.
4. Invalidate that object for the current decision without deleting the raw record.
5. Recompute the affected StateCard or candidate hypothesis from remaining valid evidence.
6. Record `rollback_to_step`, invalidated IDs, violation code and changed conclusion.
7. Retry once. A repeated violation triggers `abstain`; the deterministic fusion layer then preserves its calibrated supervised prior.

## Non-negotiable behaviour

Rollback must change the working hypothesis or explicitly justify why the final decision remains unchanged. Merely adding a warning sentence while retaining an invalid evidence contribution does not count as rollback.
