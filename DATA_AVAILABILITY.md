# Data availability

The repository contains no locally licensed participant audio recordings.
It does contain upstream-released SpeechCARE reference tables with participant
transcripts, demographic fields, labels and predictions. Those tables are not
synthetic. Their redistribution terms still require a separate publication-release
review; upstream public availability alone is not sufficient authorization.

## Public demonstration

The four WAV files under `demo/assets/synthetic_*.wav` are generated deterministically by `demo/generate_sample.py`. They are not human speech. They exercise clinical-interview, picture-description, structured-task, and natural-speech routing; feature extraction; MetricEvidence construction; StateCard aggregation; report contracts; and trace rendering. Their reference values are illustrative and must not be interpreted as clinical norms.

## Restricted research datasets

The full harness supports locally mounted copies of IAEAV, ADReSS 2020, ADReSSo 2021, PROCESS Challenge 2, PREPARE, TAUKADIAL, DementiaBank Pitt, DementiaNet public figures and NCMMSC2021 AD. Access and redistribution conditions differ by source. Users must obtain each dataset from its owner and mount it under `data/raw/` according to the matching YAML file in `configs/datasets/`.

PREPARE and TalkBank-derived recordings must not be redistributed through this repository. The repository stores configuration, schemas and code, not protected source media.

## Derived benchmark references

Files under `references/speechcare/` retain source URLs, pinned commits and intended
use. Released predictions are used for explicitly labelled retrospective
comparisons and historical output-level extensions; those extensions must not be
described as independently trained ADvoice models. They are not ground-truth
training labels. Do not add private source data or new subject-level outputs to
Git; use ignored private workspaces described in `docs/RESEARCH_WORKSPACE.md`.
