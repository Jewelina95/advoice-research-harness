# Medical scope

## Intended claim

The system estimates whether a speech sample contains patterns consistent with elevated cognitive-screening risk and whether formal cognitive assessment or repeat collection is warranted.

## Claims that are not allowed

- “Alzheimer disease is confirmed.”
- “Amyloid/tau pathology is present.”
- “This is early/middle/late AD” based only on speech.
- A causal disease mechanism inferred from an acoustic embedding, MFCC, pitch, loudness or recording artifact.
- Medication, psychiatric, hearing, neurological or respiratory diagnoses not supplied in the case context.

## Why the boundary exists

Current Alzheimer criteria distinguish clinical presentation from biological diagnosis and staging. Speech may provide a low-cost behavioural signal, but it does not measure the defining biomarkers. Dataset labels such as AD, dementia, MCI, progression or control are also not interchangeable clinical endpoints. The Agent must use the exact label definition supplied for the current task.

## Permitted clinical language

- “The sample shows/does not show speech-language findings associated with elevated screening risk.”
- “The evidence supports referral for formal cognitive assessment.”
- “The available evidence is insufficient; repeat collection is recommended.”
- “This finding may also be affected by hearing, mood, fatigue, language background, motor speech or recording conditions and requires clinical review.”

## Recommendation boundary

Recommendations may include validated cognitive screening, collateral history, functional assessment, medication/mood/hearing review, repeat standardized speech collection and specialist referral when appropriate. Do not prescribe treatment or biomarker testing as an automatic consequence of the Agent output.
