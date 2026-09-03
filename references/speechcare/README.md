# PREPARE task labels

`prepare_task_labels.csv` contains the per-UID task labels released by the
SpeechCARE authors in `our_preprocessed_train.csv`,
`our_preprocessed_validation.csv`, and `our_preprocessed_test.csv`.

- Source: https://github.com/SpeechCARE/SpeechCARE-NIA-Phase2
- Source commit: `bf1281d2e6e3617b3c81d16220a96e02646a570d`
- Retrieved: 2026-08-21
- Use: task routing only; diagnosis labels and published test predictions are
  not imported into ADvoice training.

The original released labels were normalized to seven values: picture
description, sentence reading, voice assistant, semantic verbal fluency, story
recall, personal narrative, and other.

`prepare_protocol_inputs.csv` freezes the authors' train/validation/test
partition and the accompanying transcription, task, language and age fields
from commit `25f39ae33fbb02ad8bdb8a18a2dcf0e22f66ec74`. Test diagnoses are deliberately
excluded; held-out outcomes remain evaluation-only in ADvoice.

`released_outputs/` contains the authors' public official-test probability files
from the pinned repository. They are used only in the explicitly labelled
retrospective cognition-extension experiment, never for fitting or selecting the
ADvoice cognition model. SHA-256 hashes are recorded with the experiment output.
