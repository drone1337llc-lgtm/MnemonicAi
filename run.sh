#!/usr/bin/env bash
# MnemonicAi one-click launcher (macOS / Linux).
# Installs dependencies (first run) then starts the server + live brain monitor.
#   ./run.sh                 # real model (needs ./models/ornith-1.0-9b + NVIDIA)
#   ./run.sh --mock          # try the UI with no GPU/model
set -e
cd "$(dirname "$0")"

PY=python3
command -v $PY >/dev/null 2>&1 || PY=python

# Install once (writes a marker so re-runs skip straight to start).
if [ ! -f mnemonicai_data/.installed ]; then
  echo "== First run: installing =="
  $PY install.py "$@" || true
  mkdir -p mnemonicai_data && touch mnemonicai_data/.installed
fi

echo "== Starting MnemonicAi =="
exec $PY start.py
