# CODEMAP (read this first; ~1 page instead of 34k LOC)

Package `src/advoice` (main path ~34k LOC, 45 modules + `pilot/`). Frozen one-off research code is in `src/advoice/legacy/` (~10k LOC): do not read it unless the task names a module there.

## Entry points
- CLI: `advoice` / `python -m advoice` -> `cli.py` (188 lines, thin argparse). Commands: `validate`, `run`, `run-all`, `run-processed`, `run-all-processed`, `aggregate-report`, `experiment`, `evaluate`, `evaluate-all`, `report`, `clean-cache`, `agent-led`.
- Pilot: `scripts/run_evidence_state_pilot.py` -> `src/advoice/pilot/` (contracts, data, labels, learning, runner, runtime, reporting).
- Makefile targets wrap the CLI. `scripts/` = one-off analysis/audit/report scripts (27 files); `scripts/v2/` = V2 prototype.

## Main data flow (`pipeline.py`: run_pipeline / run_processed_pipeline / run_all_*)
1. Config + data: `config.py`, `data.py`, `workspace.py`, `cache.py`, `utils.py`, `transcripts.py`, `transcript_sanitization.py`, `asr.py`.
2. Routing + features: `routing.py` -> `features.py`, `deep_embeddings.py`, `deep_audio_embeddings.py`.
3. Evidence: `evidence.py` (MetricEvidence), `evidence_replay.py`, `evidence_review.py`, `evidence_revision_*`, `metric_governance.py`.
4. States: `states.py` (StateCards), `state_graph.py`, `cognitive_prototypes.py`, `sequence_expert.py`.
5. Supervised reference: `module_a.py`, `module_b.py`, `condition_c.py` (largest, 2.4k lines; fusion/conditions), `calibration_training.py`, `dynamic_gate.py`, `decision_lock.py` (freezes numeric decision).
6. Agents: `agent_runtime.py` (provider layer: do not edit casually), `direct_agent.py`, `cognitive_agent.py`, `diagnostic_agent.py`, `diagnostic_agent_report.py`, `report_scoring_agent.py`.
7. Agent-led variant: `agent_led.py`, `agent_led_run.py`, `agent_led_study.py`, `agent_led_evaluation.py`.
8. Evaluation/reporting: `evaluation.py`, `experiments.py` (versioned recipes), `reporting.py`, `aggregate_reporting.py`, `failure_analysis.py`, `models.py` (shared dataclasses/specs).

Layering rule: numeric prediction is frozen (`decision_lock`) before any report Agent runs; reports cannot change labels.

## Configs, schemas, skills
- `configs/`: `project.yaml`, `models/default.yaml` (model hyperparameters), `agents/default.yaml` (prompts/provider), `datasets/`, `channels/`, `routes/`, `metrics/`, `states/`, `evaluation/`, `experiments/`, `calibration/`, `pilot/`, `research/`.
- `schemas/*.schema.json`: JSON contracts for evidence, decisions, routes, provenance.
- `skills/ad_evidence_diagnostic/`, `skills/ad_agent_led/`: Agent skill/contract files (e.g. `REPORT_CONTRACT.md`).
- `templates/`, `demo/`, `examples/`: report templates and synthetic demo data.

## legacy/ (frozen, run via `python -m advoice.legacy.<name>` or scripts/tests only)
authority_* (agent_bridge, joint_fusion, review_runtime, review_transaction_bridge, state_delta_cli, state_delta_study, study_dataset), conditional_authority(+_pilot), calibration_registry, condition_c_advisor, condition_c_delta, cognitive_extension, cognitive_representation_fusion, post_agent_evaluation, evidence_research(+_run), evidence_feature_audit(+_run), evidence_domain_audit_run, demo. Import as `advoice.legacy.<name>`; main path never imports them. Schema-version strings inside keep the old `advoice.<name>.` prefix on purpose (artifact provenance).

## Do NOT read by default
- `src/advoice/legacy/`, `reports/` (generated outputs), `paper/` (LaTeX/Overleaf, untracked build files), `docs/` history (dated `*_2026-*`, `EVIDENCE_GOVERNANCE_*`, `V2_*`, `LITERATURE_*`, `docs/superpowers`, `docs/reports`), `.local/` (private data/outputs), `data/`, `references/`, `*HANDOFF*.md` (historical).
- Useful docs only when needed: `docs/ARCHITECTURE.md`, `docs/AGENT_LED_ARCHITECTURE.md`, `docs/REPORT_AND_EVALUATION_SPEC.md`, `CURRENT_PROJECT_HANDOFF_2026-10-08.md`.

## Tests
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider`
Baseline: 1303 passed, 3 skipped (~1 min). Test file `tests/test_<module>.py` mirrors the module name; legacy modules are tested with the same naming.

## Known inconsistency
`condition_c.py` falls back to `minimum_training_subjects=300` for deep_audio while `configs/models/default.yaml` sets 80; the code default is left unchanged to avoid behaviour drift when configs omit the key.
