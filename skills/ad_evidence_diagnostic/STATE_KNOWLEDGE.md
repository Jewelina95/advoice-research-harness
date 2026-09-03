# Cognitive state knowledge

These states organize observable speech-language evidence. They are not diagnoses.

| ID | State | Evidence required for clinical use | Main cautions | Default role |
|---|---|---|---|---|
| S01 | Pause and fluency burden | Patient-internal pause duration/rate with role and segmentation validity | Turn gaps, prompts, VAD/ASR segmentation | reportable when reliable |
| S02 | Output efficiency | Voiced fraction, speech runs and rate; content efficiency when task scorer exists | Task duration, prompt frequency, ASR word count | reportable when reliable |
| S03 | Speech continuity | Stable fragmentation/continuity evidence across patient segments | VAD threshold and role leakage | cautious/reportable |
| S04 | Intensity stability | RMS variation after device and distance QC | Device, gain, microphone distance | model auxiliary/QC-sensitive |
| S05 | Prosodic variation | Valid F0 coverage and variation with language/sex/context considered | Pitch tracking, emotion, motor/respiratory factors | cautious/reportable |
| S06 | Low-level spectral pattern | Model representation only | Device/domain sensitivity; low clinical specificity | model-only |
| S07 | Lexical retrieval and specificity | Pronoun/generic expression, content words, word-finding pauses, fillers/repairs or naming failures | Language, task and ASR errors | reportable with convergent evidence |
| S08 | Lexical diversity | Length-controlled diversity; correlated indices must not duplicate votes | Sample length, tokenization, language | reportable when reliable |
| S09 | Semantic coherence | Validated task relevance, topic drift or discourse coherence method | Simple word overlap/embedding similarity is insufficient | unavailable until validated |
| S10 | Information density/content units | Task-specific content-unit rubric and valid transcript | No rubric, prompt differences | unavailable until validated |
| S11 | Syntactic complexity | Language-valid utterance/structure measures with adequate transcript | ASR punctuation, language grammar, task length | cautious/reportable |
| S12 | Disfluency, repetition and repair | Filled pauses, repetitions and self-repairs with language/role validity | ASR duplication and normal conversational fillers | reportable when reliable |
| S13 | Interaction/pragmatic burden | Patient turn share, support/prompt burden and response initiation with roles | Interview style and protocol | task-conditional |
| S14 | Task performance | Validated task answers/content completion | Missing rubric or nonstandard task | unavailable until validated |

## State formation rules

1. Remove unavailable metrics before fusion.
2. Do not count highly correlated measures of the same phenomenon as independent evidence.
3. Weight each remaining metric by pre-registered direction and case reliability.
4. Retain supporting and counterevidence IDs separately.
5. Mark the state `unreliable` when evidence exists but measurement quality is inadequate; mark `unavailable` when the task cannot measure it.
6. A strong model-only state can influence the supervised prior within a weight cap but cannot become a clinician-facing disease explanation.
