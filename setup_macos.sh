#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PYTHON_BIN="${POOLSTUDY_PYTHON:-python3}"
"$PYTHON_BIN" - <<'PY'
import platform,sys
if sys.version_info[:2] not in [(3,11),(3,12),(3,13)]:
    raise SystemExit('Use native Apple Silicon Python 3.11, 3.12, or 3.13. Python 3.12 is recommended.')
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise SystemExit('This Python runs under Rosetta. Install/use an arm64 Python before creating the environment.')
print('Python:',sys.version.split()[0], 'Architecture:',platform.machine())
PY
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > requirements.installed.txt
.venv/bin/python study.py doctor
printf '\nSetup finished. Activate with: source .venv/bin/activate\n'
