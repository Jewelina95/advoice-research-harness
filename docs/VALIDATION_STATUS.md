# Validation status

Status frozen on 2026-09-03.

## Completed under the 9.2 protocol

- PREPARE engineering run, method audit and SpeechCARE-aligned retrospective comparison.
- Same-encoder isolation of the cognitive representation contribution.
- Agent interface, evidence identifier, permission, fallback and calibration tests.
- Public synthetic demo and local upload API.
- 127 automated tests.

## Not completed under the 9.2 protocol

The remaining nine configured tasks have historical 8.27 artifacts but have not been rerun under 9.2. Historical artifacts must not be relabelled as 9.2 results.

The standalone PREPARE 9.2 system does not yet exceed the published SpeechCARE means on Micro AUROC, Micro F1 and Micro AUPRC. The release gate therefore remains closed. The repository records this status rather than changing thresholds or reusing held-out outcomes to manufacture a pass.

The PREPARE official test outcomes have been inspected repeatedly during development. A future model frozen after this point requires a new site, time or held-out cohort for confirmatory external validation.
