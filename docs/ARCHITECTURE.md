# Architecture

## Research pipeline

```mermaid
flowchart LR
    A[Licensed audio and transcript] --> B[Dataset adapter]
    B --> C[Task language role routing]
    C --> D[Audio text and segment features]
    D --> E[MetricEvidence]
    E --> F[Overall and task-specific StateCards]
    F --> G[Supervised out-of-fold prior]
    F --> H[Blind evidence workspace]
    H --> I[Single diagnostic Agent]
    G --> J[Validation-frozen constrained fusion]
    I --> J
    J --> K[Locked prediction]
    K --> L[Clinician and patient reports]
    E --> L
    F --> L
    D --> L
```

The Agent does not receive the supervised probability on its first evidence pass. It returns discrete evidence likelihoods and valid evidence identifiers. A validation-set gate determines whether those likelihoods may modify the supervised prior. Invalid output, insufficient evidence, confounding or failure to improve the development objective causes deterministic fallback.

## Evidence contract

```mermaid
flowchart TD
    M[Observed metric value] --> E[MetricEvidence]
    R[Training-fold reference] --> E
    Q[Audio text and role quality] --> E
    E --> S[StateCard]
    T[Task and segment identity] --> S
    S --> D[Diagnostic evidence workspace]
    D --> A[Agent evidence likelihood]
    A --> F[Constrained fusion]
    S --> P[Traceable report]
    E --> P
```

Each MetricEvidence object records the observed value, expected direction, training-only reference, reliability, missingness, confound tags, task scope, source modality and report permission. StateCards combine correlated measurements once within a clinical construct and preserve supporting and counter-evidence IDs.

## Evaluation layers

- Layer A evaluates discrimination, classification, calibration and operating points on subject-disjoint predictions.
- Layer B evaluates evidence validity, leakage controls, fallback behavior, report permission, traceability, ablations, negative controls and Agent incremental value.
- Comparisons with published work are reported only when cohort, endpoint and protocol are compatible. Retrospective comparisons are not labelled confirmatory external validation.
