#!/usr/bin/env bash
# Smoke-launch TileVision AI.app after packaging (macOS CI / local).
#
# Verifies the frozen binary starts, writes logs/config under an isolated HOME,
# and exits cleanly under a short timeout. Does NOT run the full 30-scenario
# release validation (that needs a longer GPU/CPU budget).
#
# Usage:
#   bash scripts/smoke_launch_mac_app.sh "dist/TileVision AI.app"
set -euo pipefail

APP="${1:?path to TileVision AI.app}"
BIN="$APP/Contents/MacOS/TileVisionAI"
TIMEOUT_S="${TILEVISION_SMOKE_TIMEOUT_S:-90}"

if [[ ! -x "$BIN" && ! -f "$BIN" ]]; then
  echo "ERROR: executable missing: $BIN" >&2
  exit 1
fi

SMOKE_HOME="$(mktemp -d "${TMPDIR:-/tmp}/tilevision-smoke.XXXXXX")"
cleanup() { rm -rf "$SMOKE_HOME"; }
trap cleanup EXIT

export HOME="$SMOKE_HOME"
export TILEVISION_DEV_MODE=1
export TILEVISION_OFFLINE_MODEL=1
export TILEVISION_LOG_LEVEL=INFO
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== Smoke launch ==="
echo "app:  $APP"
echo "home: $HOME"
echo "qt:   $QT_QPA_PLATFORM"
echo "timeout: ${TIMEOUT_S}s"

# Launch; kill after timeout. Exit 137/143 from kill is OK if process started.
set +e
"$BIN" >"$SMOKE_HOME/smoke_stdout.txt" 2>"$SMOKE_HOME/smoke_stderr.txt" &
PID=$!
elapsed=0
while kill -0 "$PID" 2>/dev/null; do
  if (( elapsed >= TIMEOUT_S )); then
    echo "Smoke window elapsed — terminating PID $PID"
    kill "$PID" 2>/dev/null || true
    sleep 2
    kill -9 "$PID" 2>/dev/null || true
    break
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done
wait "$PID" 2>/dev/null
RC=$?
set -e

echo "process exit/rc=$RC"
echo "--- stdout (tail) ---"
tail -40 "$SMOKE_HOME/smoke_stdout.txt" || true
echo "--- stderr (tail) ---"
tail -40 "$SMOKE_HOME/smoke_stderr.txt" || true

# Evidence of first-launch side effects under isolated HOME.
CFG="$HOME/.tilevision_ai"
echo "config dir exists: $([[ -d $CFG ]] && echo yes || echo no)"
find "$HOME" -maxdepth 4 -type f 2>/dev/null | head -40 || true

# Fail hard if the binary crashed immediately (no files / tiny stderr with Traceback).
if grep -qE 'Traceback \(most recent call last\)|ImportError|ModuleNotFoundError|Segmentation fault' \
  "$SMOKE_HOME/smoke_stderr.txt" "$SMOKE_HOME/smoke_stdout.txt" 2>/dev/null; then
  echo "ERROR: smoke launch captured a fatal Python/native error" >&2
  exit 1
fi

# At least some startup output or config should appear for a healthy launch.
if [[ ! -s "$SMOKE_HOME/smoke_stdout.txt" && ! -s "$SMOKE_HOME/smoke_stderr.txt" && ! -d "$CFG" ]]; then
  echo "ERROR: smoke launch produced no output and no config — app likely failed to start" >&2
  exit 1
fi

echo "=== Smoke launch OK ==="
