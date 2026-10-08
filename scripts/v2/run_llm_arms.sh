#!/usr/bin/env bash
# Usage: bash scripts/v2/run_llm_arms.sh <provider> <model>
#   claude:   bash scripts/v2/run_llm_arms.sh claude_cli claude-opus-5-5
#   deepseek: DEEPSEEK_API_KEY=... bash scripts/v2/run_llm_arms.sh deepseek_api deepseek-chat
set -euo pipefail
cd "$(dirname "$0")/../.."
ART="/Users/wenshaoyue/Desktop/research/ad general/AD voice/9.2/artifacts/PREPARE_DrivenData"
P="$1"; M="$2"; W="${WORKERS:-6}"
run() { python3 -W ignore scripts/v2/agent_measure.py --artifacts "$ART" --provider "$P" --model "$M" --out-dir .local/v2/agent --workers "$W" --batch "${BATCH:-12}" "$@"; }
run --arm C --limit 12            # smoke test
run --arm C &                     # full framework measurement
run --arm B &                     # full plain-agent judgement
run --arm C --ids-file .local/v2/drift_ids.txt --variant 1 &
run --arm B --ids-file .local/v2/drift_ids.txt --variant 1 &
wait
echo "done: $P $M"
