# One repository, separate private workspaces

The Git repository is the source of truth for code, configurations, contracts,
tests, and experiment protocols. Do not copy the system into a new date folder.
Keep historical artifacts read-only and create isolated run workspaces instead.

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
