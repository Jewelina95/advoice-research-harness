#!/usr/bin/env bash
# Usage: bash scripts/v2/run_multi.sh <provider> <model>   (B and C arms on the nine 8.27 diagnostic datasets)
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT="/Users/wenshaoyue/Desktop/research/ad general/AD voice/8.27/artifacts"
for ds in ADReSS_2020 ADReSSo_2021_diagnosis DementiaBank_Pitt DementiaNet_PublicFigures IAEAV NCMMSC2021_AD PROCESS_2 TAUKADIAL; do
  for arm in B C; do
    python3 -W ignore scripts/v2/agent_measure.py --artifacts "$ROOT/$ds" --arm $arm --provider "$1" --model "$2" \
      --out-dir ".local/v2/multi/$ds" --workers "${WORKERS:-4}" --batch "${BATCH:-8}" > ".local/v2/multi_${ds}_${arm}_$1.log" 2>&1 &
  done
done
wait
echo "done $1"
