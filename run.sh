#!/usr/bin/env bash
# Start Bud-E. Assumes the venv is active and the oMLX menu bar app is open.
set -e
BASE=$(python -c "from bud_e import config as C; print(C.LOCAL_BASE)")

echo "checking oMLX at $BASE ..."
curl -s "$BASE/models" >/dev/null || {
  echo "oMLX is not answering at $BASE. Open the oMLX menu bar app."; exit 1; }

wired=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if [ "${wired:-0}" -lt 40000 ]; then
  echo "note: GPU wired limit is ${wired}MB (0 = macOS default, ~75% of RAM)."
  echo "      sudo sysctl iogpu.wired_limit_mb=40960   # resets on reboot"
fi

python selfcheck.py
python -m bud_e.cli doctor
echo
python -m bud_e.cli run
