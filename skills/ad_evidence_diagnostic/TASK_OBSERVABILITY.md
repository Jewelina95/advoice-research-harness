# Task observability

Observability answers whether a task can measure a state. It is not reliability and not a learned prediction weight.

| Task family | Strongly observable states | Conditional states | Normally unavailable |
|---|---|---|---|
| Picture description | S01, S02, S07, S08, S11, S12 | S03, S05; S09/S10/S14 only with validated picture-unit scorer | S13 without interviewer roles |
| Structured clinical interview | S01, S02, S07, S08, S11, S12, S13 | S03, S05; semantic/content states when prompts and scoring are standardized | S14 without task rubric |
| Structured multi-task battery | Task-specific S01, S02, S07, S08, S11, S12; S13 with roles | S09/S10/S14 only for tasks with validated answer/content rules | Cross-task averages that erase task identity |
| Verbal fluency | Output count/rate, pause/continuity, repetitions and task-valid lexical measures | Semantic clustering/switching only with language-specific validated scorer | Picture-content units and interview burden |
| Spontaneous narrative/public speech | S01, S02, S07, S08, S11, S12 | S03/S05 after QC; S09 with validated topic-drift method | S13 and S14 without roles/rubric |
| Longitudinal follow-up | Within-person change in observable states | Diagnosis-related interpretation only with visit pairing and time context | Progression inferred from one isolated visit |

## Role rules

- Compute patient clinical metrics from patient speech only.
- Between-turn delay is not a within-utterance pause.
- Interviewer prompts and interviewer-induced waiting are interaction context, not patient pause burden.
- If role confidence is below the configured threshold, interaction states are unavailable and pause metrics are down-weighted or withheld.

## Language rules

State meaning is shared across languages, but tokenization, lexical norms, filler lists, repair patterns, task scorers and reference distributions are language-specific. A pooled cross-language reference is not acceptable when language-specific measurement is required and adequate training controls are absent.
