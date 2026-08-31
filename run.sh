#!/usr/bin/env bash
# Start everything. Assumes your venv is already activated.
set -e

echo "checking LM Studio..."
curl -sf http://localhost:1234/v1/models >/dev/null || {
  echo "LM Studio server is not running. Start it with: lms server start"; exit 1; }

python selfcheck.py
python -m bud_e.cli doctor
echo
python -m bud_e.cli run
