#!/usr/bin/env bash
# Usage: bash scripts/v2/run_multi_subset.sh <provider> <model> <dataset>...
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT="/Users/wenshaoyue/Desktop/research/ad general/AD voice/8.27/artifacts"
P="$1"; M="$2"; shift 2
for ds in "$@"; do for arm in B C; do
  python3 -W ignore scripts/v2/agent_measure.py --artifacts "$ROOT/$ds" --arm $arm --provider "$P" --model "$M" \
    --out-dir ".local/v2/multi/$ds" --workers "${WORKERS:-4}" --batch "${BATCH:-8}" > ".local/v2/multi_${ds}_${arm}_${P}.log" 2>&1 &
done; done
wait
