# Agent-led frozen-cohort studies

`agent-led-study` is the evaluation entry point for the independent Agent path.
It does not retrain a model and it never selects cases using outcome labels.

The command:

1. verifies an exact mapping between pseudonymous evidence workspaces and the
   held-out prediction cohort;
2. freezes cases by hashing the selection seed and pseudonymous case ID;
3. binds the evidence snapshot and records any available frozen supervised
   advisor components without requiring the Agent to consult them;
4. runs the Agent on the MetricEvidence/StateCard/segment evidence graph without
   exposing labels;
5. attaches truth only after inference; and
6. recomputes B1, B2, historical Ours and Agent-led results on identical cases.

Legacy non-finite values are converted to JSON null and counted in the study
manifest. Null means unobserved and cannot be interpreted as zero evidence.

Example with no external model call:

```bash
python -m advoice.cli agent-led-study \
  --dataset-id ADReSS_2020 \
  --artifact-dir /path/to/frozen/artifacts/ADReSS_2020 \
  --output-dir /path/to/new/study \
  --labels HC AD \
  --provider disabled
```

Real Agent inference requires `--provider openai_api` and the explicit
`--confirm-external-data-permission` flag. Pseudonymization does not remove all
privacy risk: evidence packages can contain transcript excerpts and derived
health information. Do not use the flag unless transfer to the configured API
has been authorized for that cohort.

The output directory is immutable and includes:

- `study_manifest.json`: source hashes, selection rule and advisor identities;
- `selected_workspaces.jsonl`: frozen inputs with no outcome labels;
- `evaluation_truth.json`: local post-inference truth mapping;
- `inference/`: complete Agent actions, decisions and exact-cohort evaluation;
- `comparison.json`: exact-case B1/B2/Ours/Agent-led metrics; and
- `study_report.html`: a concise protocol-aware report.

A SpeechCARE claim is enabled only for the complete 412-case PREPARE official
test cohort. Subsets and other datasets are marked non-comparable. Agent ranking
scores are not presented as calibrated probability AUROC.

Whether the Agent inspected the optional supervised outputs is recorded per
case. A class decision does not require that consultation: forcing it would turn
the Agent path back into post-processing for another classifier and would not
test evidence-grounded Agent judgment.
