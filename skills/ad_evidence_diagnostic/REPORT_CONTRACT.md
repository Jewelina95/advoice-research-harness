# Clinician report contract

The clinician report is rendered after the decision is locked. It cannot change class probabilities or evidence selection.

The report renderer receives structured `impression_code`, locked action,
reported class/probability or value, and a versioned template ID. Narrative text
is generated from those fields; it is not a second diagnostic inference step.
The validator rejects any mismatch with the locked decision and prohibited
biological-confirmation language.

## Required sections

1. **Screening impression**: target task, locked class/probability range, uncertainty and whether referral/retest is supported.
2. **Main observed findings**: clinically readable cognitive/speech-language states, strongest first.
3. **Traceable evidence**: for each finding show original metric, unit, training reference, direction and linked audio/transcript segment.
4. **Counterevidence**: preserved abilities or conflicting tasks that reduce certainty.
5. **Collection and interpretation limits**: clinically readable explanation of material quality/confounds.
6. **Recommended clinical review**: formal cognitive assessment, history/function, mood/medication/hearing review, repeat standardized collection or referral as justified.

## Trace map

Every risk or impairment statement must follow:

`report statement -> locked risk contribution -> StateCard -> MetricEvidence -> task/segment -> source audio/transcript`

## Citation fields

- `used_evidence_ids`: reportable clinical support only.
- `counterevidence_ids`: reportable clinical counterevidence only.
- `quality_evidence_ids`: acquisition/processing limitations only.

Do not display raw embedding dimensions, internal prompt text, model chain-of-thought or source diagnosis identifiers.
