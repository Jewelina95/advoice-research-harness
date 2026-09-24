# Task Package Review

Date: 2026-09-24
Scope: execution-plan coherence, not proof of implemented model performance.
Status: plan ready for dispatch; implementation and paid pilot have NOT started.

An independent Agent reviewed the package and identified six execution defects. All six were revised and rechecked:

| Finding | Resolution |
|---|---|
| Implementation and paid execution shared one dependency node | Explicit T6A -> T7P -> T6B -> T7R nodes |
| Two-call subject limit conflicted with paired model experiment | Core per-subject cap separated from per-model/representation comparison cap |
| Source ownership omitted permitted local artifacts | Separate write_files and artifact_write_paths; coordinator-only Git operations |
| Label isolation was only described, not enforced | Stage capability manifests and provider-boundary validation required |
| Final refit could contaminate calibration inputs | OOF/calibrator freeze precedes final refit; final-fit IDs rejected from calibration |
| Canary exclusion and fallback unspecified | Four disjoint partitions; 281 development, 70 holdout, 20 stress; reduce canary if insufficient |

Independent final response: no residual blocker among these six findings.

Local mechanical checks passed:
- All ten dispatch nodes reference existing task documents.
- Dependency graph is acyclic; only T6B can make paid calls after T7P.
- Tracked source write ownership is unique.
- Paired-model bound: 24 * 2 * (2 + 1) = 144 semantic calls.
- Conservative semantic bound: 742 + 144 + 24 = 910; transport attempts are separately capped.
- git diff --check passed.

No runtime source was changed for this task package. Existing uncommitted user/code work remains untouched. New functions, CLI and tests described in the package are implementation requirements, not verified existing functionality. Scientific validity, actual costs, calibration quality and performance improvement must be checked during implementation and pilot.
