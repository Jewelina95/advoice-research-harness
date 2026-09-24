# T5: Render Fixed Reports from Locked Results

Model: gpt-5.6-luna; reasoning: medium.
Depends on: T1, T2, T3.
Read START_HERE.md and CONTRACTS.md first. All repository-relative paths resolve from the assigned worktree root.
Write ownership: `src/advoice/pilot/reporting.py`, `templates/pilot/system_report.html.j2`, `templates/pilot/evaluation_report.html.j2`, `tests/test_pilot_reporting.py`.
All source code and public UI text must be English. Do not launch unrelated work.

## Inputs
Shared records, docs/REPORT_AND_EVALUATION_SPEC.md, existing repository templates and 8.27/reports/latest/evaluation_report.html for visual conventions only. Use current system semantics, not the old code.

## Steps
- [ ] Reuse chart palette/layout helpers where available. Read aggregate result JSON, never refit models or resample subjects.
- [ ] System HTML: exact implemented pipeline, dataset inputs, trained vs fixed components, a de-identified trace example, versions and unresolved limits.
- [ ] Evaluation HTML: separate Layer A summary and Layer B summary, followed by single-column detail figures; all plot labels English with legible horizontal axes and annotated values.
- [ ] Layer A: per-dataset accuracy, balanced accuracy, macro F1, macro/micro AUROC when defined, per-class sensitivity/specificity, calibration, CIs, confusion matrices and help/harm counts.
- [ ] Layer B: citation validity, trace completeness, accepted/rejected repair, counterfactual consistency, fallback, API failure, coverage and actual tokens/cost. These are technical proxies, not physician-rated clinical safety.
- [ ] Show each metric's denominator, undefined reason and sample count. Separate stress results from trained pilot results; never pool them into a headline clinical average.
- [ ] Show B_raw/B/J-A/J-S/J-AS consistently. Historical scores labeled historical, not paired improvements. SpeechCARE superiority section remains not evaluated until matching formal protocol artifacts exist.
- [ ] No per-case clinician report generation. No dataset labels or raw transcript content in publishable fixtures.
- [ ] Validate empty/failing runs produce an honest incomplete report rather than charts with fake zeros.

## Acceptance
```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q tests/test_pilot_reporting.py
```
Expected: fixture output includes Layer A, Layer B and all failures; numbers exactly match aggregate JSON. Inspect rendered desktop/mobile screenshot for clipping; record screenshot paths. New plot statistics must be computed upstream by T6, not hidden in templates.

## Required completion response
Write .local/pilot-handoff/T5/WORKER_RESULT.md with: base SHA, final SHA, changed files, exact test commands/results, artifact paths, remaining findings, schema/config changes requested, paid calls and tokens (zero for code-only tasks). Return this path and a summary under 400 words. Do not paste patient data or the entire code diff into coordinator chat.
