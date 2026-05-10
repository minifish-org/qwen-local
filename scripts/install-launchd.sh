#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="org.minifish.qwen4b-local"
SRC="$ROOT/launchd/${LABEL}.plist"
DST="$HOME/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"

if [[ ! -f "$SRC" ]]; then
  echo "Missing plist template: $SRC" >&2
  exit 1
fi

if [[ ! -x "$ROOT/scripts/run-server.sh" ]]; then
  echo "Missing executable server script: $ROOT/scripts/run-server.sh" >&2
  exit 1
fi

if [[ ! -x "$ROOT/llama.cpp/build/bin/llama-server" ]]; then
  echo "llama-server is not built yet. Run ./scripts/setup-llama.sh first." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"
cp "$SRC" "$DST"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
fi

launchctl bootstrap "$DOMAIN" "$DST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL"

echo "Installed and started $LABEL"
echo "Logs:"
echo "  $ROOT/logs/launchd.out.log"
echo "  $ROOT/logs/launchd.err.log"
