# One repository, separate private workspaces

The Git repository is the source of truth for code, configurations, contracts,
tests, and experiment protocols. Do not copy the system into a new date folder.
Keep historical artifacts read-only and create isolated run workspaces instead.

## Recommended single entry point

Use `main` from the GitHub repository as the maintained framework. The code is a
continuation of the 9.2 system with reviewed engineering fixes, not a new clinical
model or a claim that benchmark superiority has been achieved. Pull updates before
starting an experiment, never while it is running. Do not edit historical source
copies in the surrounding dated directories.

Create a private recipe using `configs/experiments/raw.example.yaml` or
`configs/experiments/processed.example.yaml` as its schema. The recipe lists the
registered datasets, local input directories, output workspace, run mode, optional
complete model configuration, and explicit Agent provider. Relative paths resolve
against the recipe file, not the current shell directory. Never put API keys there.

```bash
make experiment-check CONFIG=/absolute/path/to/private/experiment.yaml
make experiment CONFIG=/absolute/path/to/private/experiment.yaml
```

Equivalent installed CLI: `advoice experiment --config /path/to/experiment.yaml`.
The check command verifies configuration and input presence only. It does not
validate clinical quality, train models, or invoke an API. `agent_provider` defaults
to `disabled`; API-backed experiments require an explicit `openai_api` selection
and credentials in the environment. Choosing the provider does not bypass the
clinical correction gates described in `AGENT_CALIBRATION_SAFETY.md`.

Each execution records Git commit/branch/dirty status, source/config fingerprints,
the resolved recipe, Python runtime, per-dataset logs and new immutable run IDs.
A shared workspace lock covers both managed execution and the older CLI commands.
Only a child with the current invocation's lock token can use its parent's lock.
A crash may leave a lock;
inspect its PID and verify the process has stopped before removing it. Do not run
uncoordinated direct Python API calls against that workspace. Output paths and
existing descendants are checked for symlinks before writes.

A dataset is successful only if the pipeline publishes a new run and its system
report, evaluation report, Layer A PNG and Layer B PNG all exist. Aggregate figures
are generated only for datasets that succeeded in this invocation, not arbitrary
older artifacts. Any failed dataset returns a nonzero overall exit code. A partial
aggregate is not a passed experiment. Source/config changes during execution also
fail the frozen-version check. Failures remain in `experiment_latest.json` and the
execution's own `status.json`; an old HTML on disk is not evidence of a new success.

| Output | Location under the private workspace |
| --- | --- |
| Latest execution status | `experiment_latest.json` |
| Execution history and logs | `executions/<id>/` |
| Immutable aggregate inputs, reports and A/B figures | `executions/<id>/aggregate/` |
| Per-dataset immutable reports and artifacts | `runs/<run_id>/` |
| Current per-dataset system/evaluation reports and A/B figures | `reports/datasets/<dataset>/latest/` |
| Current selected-cohort system/evaluation reports and A/B figures | `reports/latest/` |

The aggregate reads copies of the captured immutable dataset runs, not mutable
current artifacts. Its archived reports and hashes remain attached to the
execution. The current aggregate is a convenience copy replaced on the next
successful aggregation. Per-dataset run reports remain versioned. Keep separate workspaces
for experiments with different scientific protocols. The runner reuses existing
pipeline stage caches; it does not redefine metrics, silently change splits, or
substitute a synthetic demo for a trained cohort experiment.

To add recordings to an existing dataset, update its local data and rerun the same
recipe. To add a genuinely new dataset, first add/review its versioned adapter,
task/label/speaker/split configuration and leakage tests, then list its ID in the
recipe. A new folder alone cannot safely determine those semantics automatically.

## Layout

| Location | Contents | Versioned |
| --- | --- | --- |
| `src/advoice` | Executable pipeline | Yes |
| `configs`, `schemas`, `skills` | Model, evidence, Agent and report contracts | Yes |
| `tests` | Unit and regression tests | Yes |
| `scripts` | Explicit experimental entry points | Yes |
| `docs` | Protocol, decisions and limitations | Yes |
| `demo` | Synthetic public demonstration | Yes |
| `.local/<experiment>` | Private inputs, predictions, models and reports | No |
| External raw-data directory | Licensed recordings and transcripts | No |

The public demo is not the trained diagnostic pipeline. Its fixtures must not
be used to claim predictive performance. Historical results must retain their
original run identity and protocol.

## Isolated local run

```bash
export ADVOICE_RAW_DATA_DIR="/path/to/licensed/raw-data"
export ADVOICE_WORKSPACE_DIR="$PWD/.local/my-experiment"
make validate DATASET=NCMMSC2021_AD
make full DATASET=NCMMSC2021_AD
```

Code and default configuration remain in the repository. Artifacts, immutable
run snapshots and reports go into the workspace. An optional
`ADVOICE_MODEL_CONFIG=/absolute/path/to/model.yaml` selects a complete model
configuration and records its hash in the run manifest. Use different workspaces
for different configurations; do not concurrently write the same dataset folder.

To reuse explicitly frozen preprocessing and baseline outputs:

```bash
PYTHONPATH=src .venv/bin/python -m advoice run-processed \
  --dataset ADReSSo_2021_progression \
  --source-root /path/to/frozen/artifacts --agent-provider disabled
```

This retrains condition C, not the encoders, B1, B2, or ASR. With Agent disabled,
report scoring is also disabled. Do not describe it as a live GPT experiment.
For a batch, use `run-all-processed --source-root ... --datasets DATASET1 DATASET2`.

`make all-full` executes datasets independently of the SpeechCARE performance
target. `make prepare-release-gate` is a separate benchmark claim check, not a
prerequisite for unrelated experiments.

## Experimental progression

1. Run regression tests and inspect input manifests and participant identities.
2. Run a small, prespecified cross-task pilot with isolated outputs.
3. Fix technical failures, rerun the same pilot, and retain both logs.
4. Select models on training/development data only; lock settings before test.
5. Run all eligible datasets. Report failures and exclusions rather than dropping them.
6. Publish aggregate figures and reproducibility metadata, not participant data.

Tests establish implementation properties, not model superiority. A positive
metric difference is not necessarily statistically reliable. The existing PREPARE
test set has already been inspected during development, so new results on it are
retrospective and need fresh external confirmation.

## Inventory without retraining

```bash
PYTHONPATH=src .venv/bin/python scripts/inventory_experiments.py \
  --source /path/to/historical/artifacts \
  --source "$ADVOICE_WORKSPACE_DIR/artifacts" \
  --output "$ADVOICE_WORKSPACE_DIR/reports/inventory"
```

Use a Git branch for each bounded change, attach tests and run identifiers to
the pull request, and merge only after checking scientific and engineering claims.
