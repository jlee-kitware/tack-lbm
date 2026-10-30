#!/bin/bash
# Start the Tack LBM solver (Catalyst Live on :22222) and the CFD assistant
# (trame on :8080) as user services. Extra arguments go to run_lbm.py.
#   ./run_live.sh --collision les --obstacle cylinder --re 300
#   CASE=cht ./run_live.sh --power 50       (heat sink, conjugate heat transfer with PyMFEM)
# Both services stop after RUN_LIMIT (default 2h); RUN_LIMIT=infinity disables it.
set -e
RUN_LIMIT=${RUN_LIMIT:-2h}
HERE=$(cd "$(dirname "$0")" && pwd)
CASE=${CASE:-cylinder}
if [ "$CASE" = cht ]; then
  DRIVER=run_cht.py
  PYTHON=${PYTHON:-$HERE/.venv-cpu/bin/python}
else
  DRIVER=run_lbm.py
  PYTHON=${PYTHON:-$HERE/.venv/bin/python}
fi
PV_BUILD=${PV_BUILD:-$HOME/gitsync/paraview_build_cuda}
CFD=${CFD_APP:-$HOME/gitsync/trame-cfd-mcp/examples/cfd_assistant}
LOGS=${LOGS:-$HOME/gitsync/tools}

systemctl --user stop tack-lbm cfd-app 2>/dev/null || true
systemctl --user reset-failed tack-lbm cfd-app 2>/dev/null || true

. "$HERE/env.sh"
systemd-run --user --unit tack-lbm --working-directory="$HERE" -p RuntimeMaxSec="$RUN_LIMIT" \
  --setenv=PYTHONPATH="$PYTHONPATH" --setenv=LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  -p StandardOutput=truncate:"$LOGS/tack_lbm.log" -p StandardError=inherit \
  "$PYTHON" -u "$DRIVER" "$@"

systemd-run --user --unit cfd-app --working-directory="$CFD" -p RuntimeMaxSec="$RUN_LIMIT" \
  --setenv=TRAME_DEFAULT_HOST=0.0.0.0 \
  --setenv=CATALYST_LIVE_PROXIES="$HERE/steering.xml" \
  --setenv=LD_LIBRARY_PATH="$HOME/gitsync/anari_install/lib" \
  -p StandardOutput=truncate:"$LOGS/app_8080.log" -p StandardError=inherit \
  "$PV_BUILD/bin/pvpython" --force-offscreen-rendering --venv .venv cfd_chat.py --live --server --timeout=0

echo "solver log: $LOGS/tack_lbm.log (both stop after $RUN_LIMIT; stop now: systemctl --user stop tack-lbm cfd-app)"
echo "app: http://$(hostname -I | awk '{print $1}'):8080/"
