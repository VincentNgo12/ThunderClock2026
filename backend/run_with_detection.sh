#!/usr/bin/env sh
# Start the telemetry backend with the fall-detection hub attached (detection/hub/demo_app.py, live mode).
# The backend app is unchanged; the hub wraps it, receives every raw IMU frame through the backend's raw
# hand-off and publishes fall_state / detector in the same WebSocket snapshot.
#
# Default: TELEMETRY_SOURCE=manual, so it waits for "Connect" (spine frontend or pitch deck). Detection
# starts by itself as soon as the vest streams raw frames: connect with "Quaternions + raw" (both) or
# "Raw, fused on laptop" (raw_fusion). Plain "Quaternions" carries no raw data, so the detector stays idle.
#
#   ./run_with_detection.sh                                   # http://127.0.0.1:8000, Connect from a page
#   TELEMETRY_SOURCE=hardware DEVICE_FORMAT=raw_fusion ./run_with_detection.sh   # connect at startup
#   PORT=8010 ./run_with_detection.sh
#
# Uses detection/simulation/.venv (numpy, scikit-learn, fastapi, bleak); the backend's own .venv lacks the
# detector's packages. Prototype indicator, not a medical or safety-certified device.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/detection/simulation/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Missing $PY. Create the simulation venv first (see detection/simulation/README.md)." >&2
  exit 1
fi
TELEMETRY_SOURCE="${TELEMETRY_SOURCE:-manual}"
case "$TELEMETRY_SOURCE" in
  hardware|manual) ;;
  *) echo "TELEMETRY_SOURCE must be manual or hardware here; for simulated falls run detection/hub/demo_app.py (see its header)." >&2; exit 1 ;;
esac
export TELEMETRY_SOURCE
export HUB_DEMO=1
cd "$ROOT/detection"
PYTHONPATH=".:simulation:../backend${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$PY" -m uvicorn hub.demo_app:app --host 127.0.0.1 --port "${PORT:-8000}"
