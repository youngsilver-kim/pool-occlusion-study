#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  printf 'First run: bash setup_macos.sh\n' >&2
  exit 1
fi
# caffeinate is scoped to this process and ends when Python exits.
exec caffeinate -i .venv/bin/python study.py all --config configs/main.json
