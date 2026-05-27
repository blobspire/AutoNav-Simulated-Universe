#!/usr/bin/env bash
# Run the standalone LiDAR RSSI line-detection simulator.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
SIM="$ROOT/simulated_world"
PY="$SIM/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "[ERROR] Python venv not found at:"
    echo "  $PY"
    echo "Run uv venv inside simulated_world/ and install requirements.txt."
    read -n 1 -s -r -p "Press any key to continue..."
    echo
    exit 1
fi

cd "$SIM"
"$PY" "$SIM/lidar_line_sim.py" "$@"
status=$?
if [ "$status" -ne 0 ]; then
    read -n 1 -s -r -p "Press any key to continue..."
    echo
fi
exit "$status"
