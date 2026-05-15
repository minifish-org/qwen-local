#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOMAIN="gui/$(id -u)"
LABEL="org.minifish.qwen-local"
LEGACY_LABEL="org.minifish.qwen-local-embedding"
SRC="$ROOT/launchd/${LABEL}.plist"
DST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [[ ! -x "$ROOT/scripts/run-server.sh" ]]; then
  echo "Missing executable server script: $ROOT/scripts/run-server.sh" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"
cp "$SRC" "$DST"

if launchctl print "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || true
fi
rm -f "$HOME/Library/LaunchAgents/${LEGACY_LABEL}.plist"

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
