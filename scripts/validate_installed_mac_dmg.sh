#!/usr/bin/env bash
# Install TileVision AI from a customer DMG into /Applications and run the
# Release Validation Suite against the INSTALLED packaged binary.
#
# Does NOT run `python qa_e2e/...` from a source venv as the product under test.
# The product under test is:
#   /Applications/TileVision AI.app/Contents/MacOS/TileVisionAI
#
# Usage:
#   bash scripts/validate_installed_mac_dmg.sh \
#       dist/TileVision-AI-AppleSilicon.dmg \
#       /path/to/checkout \
#       /path/to/out
#
# Env:
#   TILEVISION_RELEASE_PROFILE=pr|full
#   TILEVISION_QA_TILES=4
set -euo pipefail

DMG="${1:?DMG path required}"
SUITE_DIR="${2:?path to repo checkout containing qa_e2e required}"
OUT_DIR="${3:?output artifact directory required}"
PROFILE="${TILEVISION_RELEASE_PROFILE:-pr}"

APP_NAME="TileVision AI.app"
INSTALL_PATH="${TILEVISION_INSTALL_PATH:-/Applications/${APP_NAME}}"
BIN="${INSTALL_PATH}/Contents/MacOS/TileVisionAI"

if [[ ! -f "$DMG" ]]; then
  echo "ERROR: DMG not found: $DMG" >&2
  exit 1
fi
if [[ ! -d "$SUITE_DIR/qa_e2e" ]]; then
  echo "ERROR: qa_e2e not found under suite dir: $SUITE_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
DIAG_DIR="$OUT_DIR/macos_diagnostics"
LOG_DIR="$OUT_DIR/installed_app_logs"
mkdir -p "$DIAG_DIR" "$LOG_DIR"
LOG="$OUT_DIR/packaged_validation.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Packaged app release validation ==="
echo "dmg:    $DMG"
echo "suite:  $SUITE_DIR"
echo "out:    $OUT_DIR"
echo "profile:$PROFILE"
echo "host:   $(uname -a)"
date -u

MOUNT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tv-dmg-mount.XXXXXX")"
ATTACHED_DEV=""
SAMPLE_PID=""
cleanup() {
  if [[ -n "${SAMPLE_PID:-}" ]]; then
    kill "$SAMPLE_PID" >/dev/null 2>&1 || true
    wait "$SAMPLE_PID" 2>/dev/null || true
  fi
  if [[ -n "$ATTACHED_DEV" ]]; then
    hdiutil detach "$ATTACHED_DEV" -force >/dev/null 2>&1 || true
  fi
  if [[ -d "$MOUNT_DIR" ]]; then
    hdiutil detach "$MOUNT_DIR" -force >/dev/null 2>&1 || true
    rmdir "$MOUNT_DIR" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[1/7] Mounting DMG..."
ATTACH_OUT="$(hdiutil attach -nobrowse -readonly -mountpoint "$MOUNT_DIR" "$DMG")"
echo "$ATTACH_OUT" | tee "$DIAG_DIR/hdiutil_attach.txt"
ATTACHED_DEV="$(echo "$ATTACH_OUT" | awk '/\/dev\// {print $1; exit}')"

SRC_APP="$(find "$MOUNT_DIR" -maxdepth 2 -name '*.app' -type d | head -n 1 || true)"
if [[ -z "$SRC_APP" || ! -d "$SRC_APP" ]]; then
  echo "ERROR: no .app found inside DMG" >&2
  ls -la "$MOUNT_DIR" | tee "$DIAG_DIR/dmg_listing.txt" || true
  exit 1
fi
echo "found app in DMG: $SRC_APP" | tee "$DIAG_DIR/dmg_app_path.txt"
ls -laR "$SRC_APP/Contents" > "$DIAG_DIR/dmg_app_contents.txt" 2>&1 || true

install_app_to() {
  local dest="$1"
  echo "Installing to $dest ..."
  if [[ -d "$dest" ]]; then
    rm -rf "$dest" 2>/dev/null || sudo rm -rf "$dest" || return 1
  fi
  mkdir -p "$(dirname "$dest")" || return 1
  if ! ditto "$SRC_APP" "$dest" 2>"$DIAG_DIR/install_error.txt"; then
    echo "ditto failed — retrying with sudo"
    cat "$DIAG_DIR/install_error.txt" || true
    sudo rm -rf "$dest" || true
    if ! sudo ditto "$SRC_APP" "$dest"; then
      return 1
    fi
    sudo xattr -cr "$dest" || true
  else
    xattr -cr "$dest" || true
  fi
  return 0
}

echo "[2/7] Installing to customer Applications path..."
if ! install_app_to "$INSTALL_PATH"; then
  echo "WARN: /Applications install failed — falling back to ~/Applications"
  INSTALL_PATH="${HOME}/Applications/${APP_NAME}"
  BIN="${INSTALL_PATH}/Contents/MacOS/TileVisionAI"
  install_app_to "$INSTALL_PATH"
fi
echo "installed_path=${INSTALL_PATH}" | tee "$DIAG_DIR/installed_path.txt"
xattr -l "$INSTALL_PATH" > "$DIAG_DIR/xattr_after_clear.txt" 2>&1 || true

if [[ ! -f "$BIN" ]]; then
  echo "ERROR: installed binary missing: $BIN" >&2
  exit 1
fi
echo "installed binary: $BIN"
file "$BIN" | tee "$DIAG_DIR/installed_binary_file.txt"
ls -lh "$BIN" | tee "$DIAG_DIR/installed_binary_ls.txt"
plutil -p "${INSTALL_PATH}/Contents/Info.plist" > "$DIAG_DIR/installed_info_plist.txt" 2>&1 || true
du -sh "$INSTALL_PATH" | tee "$DIAG_DIR/installed_app_size.txt"

echo "[3/7] Detaching DMG (installed copy is under test)..."
hdiutil detach "$ATTACHED_DEV" -force >/dev/null 2>&1 || hdiutil detach "$MOUNT_DIR" -force >/dev/null 2>&1 || true
ATTACHED_DEV=""
echo "detached" | tee "$DIAG_DIR/hdiutil_detach.txt"

echo "[4/7] Preflight --verify-bundle on INSTALLED app..."
# Prefer packaged (frozen) modules. Do not let checkout src shadow the .app.
unset PYTHONPATH || true
export TILEVISION_DEV_MODE=1
export TILEVISION_OFFLINE_MODEL=1
export TILEVISION_QA_SUITE_DIR="$SUITE_DIR"
export TILEVISION_QA_PACKAGED_APP=1
export TILEVISION_QA_OUT="$OUT_DIR"
export TILEVISION_RELEASE_PROFILE="$PROFILE"
export TILEVISION_QA_TILES="${TILEVISION_QA_TILES:-4}"
export TILEVISION_QA_INDEX_TIMEOUT="${TILEVISION_QA_INDEX_TIMEOUT:-3600}"
export TILEVISION_QA_SEARCH_STALL_S="${TILEVISION_QA_SEARCH_STALL_S:-180}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

set +e
"$BIN" --verify-bundle > "$OUT_DIR/verify_bundle.txt" 2> "$OUT_DIR/verify_bundle.stderr.txt"
VERIFY_RC=$?
set -e
cat "$OUT_DIR/verify_bundle.txt" || true
cat "$OUT_DIR/verify_bundle.stderr.txt" || true
echo "verify-bundle exit=$VERIFY_RC" | tee "$DIAG_DIR/verify_bundle_rc.txt"
if [[ $VERIFY_RC -ne 0 ]]; then
  echo "ERROR: --verify-bundle failed on installed app" >&2
  exit 1
fi
if ! grep -q "bundle OK" "$OUT_DIR/verify_bundle.txt" 2>/dev/null; then
  # Windowed builds may swallow stdout; exit 0 is still required evidence of launch.
  echo "WARN: verify-bundle stdout missing 'bundle OK' (console=False possible); exit code was 0"
fi

echo "[5/7] Running full Release Validation Suite via INSTALLED binary..."
# Continuous CPU/RSS samples while the suite runs
(
  while true; do
    {
      date "+%Y-%m-%dT%H:%M:%S%z"
      ps -ax -o pid,%cpu,%mem,rss,vsz,etime,command | grep -i "[T]ileVision" || true
      vm_stat 2>/dev/null | head -20 || true
      echo "---"
    } >> "$DIAG_DIR/resource_samples.txt"
    sleep 30
  done
) &
SAMPLE_PID=$!

set +e
"$BIN" --release-validation --profile "$PROFILE" --out "$OUT_DIR" \
  > "$OUT_DIR/release_validation.stdout.txt" \
  2> "$OUT_DIR/release_validation.stderr.txt"
RC=$?
set -e
echo "release-validation exit=$RC" | tee "$DIAG_DIR/release_validation_rc.txt"
tail -80 "$OUT_DIR/release_validation.stdout.txt" || true
tail -80 "$OUT_DIR/release_validation.stderr.txt" || true

kill "$SAMPLE_PID" >/dev/null 2>&1 || true
wait "$SAMPLE_PID" 2>/dev/null || true
SAMPLE_PID=""

echo "[6/7] Collecting macOS diagnostics / crash / console logs..."
# Crash / diagnostic reports
cp -R "$HOME/Library/Logs/DiagnosticReports/"*TileVision* "$DIAG_DIR/" 2>/dev/null || true
cp -R "$HOME/Library/Logs/DiagnosticReports/"*tilevision* "$DIAG_DIR/" 2>/dev/null || true
cp -R "$HOME/Library/Logs/DiagnosticReports/"*TileVisionAI* "$DIAG_DIR/" 2>/dev/null || true
# Unified / system log snippets (best-effort; may be empty without privileges)
if command -v log >/dev/null 2>&1; then
  log show --last 2h --predicate 'process CONTAINS "TileVision" OR processImagePath CONTAINS "TileVision"' \
    --style compact > "$DIAG_DIR/macos_console_tilevision.txt" 2>/dev/null || true
fi
# App support / legacy data from the validation HOME if present
if [[ -d "$HOME/.tilevision_ai/logs" ]]; then
  cp -R "$HOME/.tilevision_ai/logs/." "$LOG_DIR/" 2>/dev/null || true
fi
# Isolated QA homes created under OUT_DIR / tmp may also hold logs — copy any found
find /tmp /var/folders "$OUT_DIR" -maxdepth 6 -type d -name ".tilevision_ai" 2>/dev/null \
  | head -20 \
  | while read -r d; do
      mkdir -p "$LOG_DIR/$(basename "$(dirname "$d")")"
      cp -R "$d/logs/." "$LOG_DIR/$(basename "$(dirname "$d")")/" 2>/dev/null || true
    done || true

{
  echo "=== host ==="
  uname -a
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  echo "=== memory ==="
  vm_stat 2>/dev/null || true
  echo "=== process sample ==="
  ps aux | grep -i TileVision | grep -v grep || true
  echo "=== installed path ==="
  echo "$INSTALL_PATH"
  echo "=== frozen evidence ==="
  echo "TILEVISION_QA_PACKAGED_APP=${TILEVISION_QA_PACKAGED_APP:-}"
} >"$DIAG_DIR/macos_system_snapshot.txt" || true

echo "[7/7] Enforcing packaged verdict..."
SUMMARY="$OUT_DIR/release_summary.json"
if [[ ! -f "$SUMMARY" ]]; then
  echo "ERROR: missing release_summary.json — packaged validation did not finish" >&2
  exit 1
fi

python3 - "$SUMMARY" "$INSTALL_PATH" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
installed = sys.argv[2]
data = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps(data, indent=2))
verdict = data.get("verdict")
frozen = data.get("frozen")
packaged = data.get("packaged_app")
exe = str(data.get("executable") or "")
print(f"PACKAGED_APP_VERDICT={verdict}")
print(f"frozen={frozen} packaged_app={packaged} executable={exe}")
print(f"installed_path={installed}")
ok = verdict == "PASS" and bool(frozen) and bool(packaged)
if not ok:
    reasons = []
    if verdict != "PASS":
        reasons.append(f"verdict={verdict}")
    if not frozen:
        reasons.append("not frozen (source-tree run)")
    if not packaged:
        reasons.append("packaged_app flag missing")
    print("FAIL reasons:", "; ".join(reasons))
    sys.exit(1)
sys.exit(0)
PY
RC2=$?

if [[ "$RC" -ne 0 || "$RC2" -ne 0 ]]; then
  echo "FAIL: packaged application did not pass release validation"
  exit 1
fi

echo "PASS: packaged application release validation succeeded"
echo "installed_app=${INSTALL_PATH}"
exit 0
