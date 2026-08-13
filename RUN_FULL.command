#!/bin/zsh
set -euo pipefail
ROOT="${0:A:h}"
cd "$ROOT"
make full
open "$ROOT/reports/latest/index.html"

