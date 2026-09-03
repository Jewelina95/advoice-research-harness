# Label Leakage Policy

Version: `label-leakage-1.0`

The prediction package must be label-free. This applies to fields, filenames,
metadata, transcripts, audio prompts and reference statistics.

## Segment handling

Every source segment is annotated with `diagnostic_disclosure` and
`prediction_eligible`.

- `none`: no explicit diagnosis or study label is present.
- `participant_self_report`: the participant states a diagnosis or diagnostic history.
- `interviewer_or_metadata`: an interviewer, transcript header or metadata reveals diagnosis or study class.
- `uncertain`: disclosure cannot be excluded.

Any value other than `none` forces `prediction_eligible=false`. The segment may
be retained for provenance or quality audit, but no clinical-support metric,
model-auxiliary feature, pretrained embedding or StateCard may cite it for prediction. Automated multilingual phrase detection
is only a screening layer; adapters must also parse dataset-specific transcript
headers and speaker roles.

## Reference handling

Reference statistics are represented by a training-only artifact ID, fold ID
and population type. The ID must resolve in the trusted artifact registry for
the same run/config/data/Skill hashes; the registry stores the immutable
artifact hash. Free-text reference names are display labels only. Test,
held-out or full-dataset statistics are prohibited. The data adapter must create
reference artifacts only after the outer split is frozen.

The held-out label is available only to the evaluation process after the locked
decision and report are written. It never enters the case evidence package,
Agent tools or cognitive trace.
