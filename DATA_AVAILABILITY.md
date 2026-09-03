# Data availability

The repository intentionally contains no participant recording, protected transcript, clinical label, or identifiable metadata.

## Public demonstration

The four WAV files under `demo/assets/synthetic_*.wav` are generated deterministically by `demo/generate_sample.py`. They are not human speech. They exercise clinical-interview, picture-description, structured-task, and natural-speech routing; feature extraction; MetricEvidence construction; StateCard aggregation; report contracts; and trace rendering. Their reference values are illustrative and must not be interpreted as clinical norms.

## Restricted research datasets

The full harness supports locally mounted copies of IAEAV, ADReSS 2020, ADReSSo 2021, PROCESS Challenge 2, PREPARE, TAUKADIAL, DementiaBank Pitt, DementiaNet public figures and NCMMSC2021 AD. Access and redistribution conditions differ by source. Users must obtain each dataset from its owner and mount it under `data/raw/` according to the matching YAML file in `configs/datasets/`.

PREPARE and TalkBank-derived recordings must not be redistributed through this repository. The repository stores configuration, schemas and code, not protected source media.

## Derived benchmark references

Files under `references/speechcare/` retain source URLs, pinned commits and intended use. Released predictions are used only for explicitly labelled retrospective comparisons; they are never treated as training labels for ADvoice.
