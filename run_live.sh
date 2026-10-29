#!/bin/bash
# Start the Tack LBM solver (Catalyst Live on :22222) and the CFD assistant
# (trame on :8080) as user services. Extra arguments go to run_lbm.py.
#   ./run_live.sh --collision les --obstacle cylinder --re 300
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PV_BUILD=${PV_BUILD:-$HOME/gitsync/paraview_build_cuda}
CFD=${CFD_APP:-$HOME/gitsync/trame-cfd-mcp/examples/cfd_assistant}
LOGS=${LOGS:-$HOME/gitsync/tools}

systemctl --user stop tack-lbm cfd-app 2>/dev/null || true
systemctl --user reset-failed tack-lbm cfd-app 2>/dev/null || true

. "$HERE/env.sh"
systemd-run --user --unit tack-lbm --working-directory="$HERE" \
  --setenv=PYTHONPATH="$PYTHONPATH" --setenv=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  -p StandardOutput=truncate:"$LOGS/tack_lbm.log" -p StandardError=inherit \
  "$HERE/.venv/bin/python" -u run_lbm.py "$@"

systemd-run --user --unit cfd-app --working-directory="$CFD" --setenv=TRAME_DEFAULT_HOST=0.0.0.0 \
  --setenv=CATALYST_LIVE_PROXIES="$HERE/steering.xml" \
  --setenv=LD_LIBRARY_PATH="$HOME/gitsync/anari_install/lib" \
  -p StandardOutput=truncate:"$LOGS/app_8080.log" -p StandardError=inherit \
  "$PV_BUILD/bin/pvpython" --force-offscreen-rendering --venv .venv cfd_chat.py --live --server --timeout=0

echo "solver log: $LOGS/tack_lbm.log"
echo "app: http://$(hostname -I | awk '{print $1}'):8080/"
