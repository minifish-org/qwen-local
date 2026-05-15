#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOMAIN="gui/$(id -u)"
LABEL="org.minifish.qwen-local"
LEGACY_LABEL="org.minifish.qwen-local-embedding"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LEGACY_PLIST="$HOME/Library/LaunchAgents/${LEGACY_LABEL}.plist"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
fi

if launchctl print "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || true
fi

rm -f "$PLIST" "$LEGACY_PLIST"

echo "Uninstalled $LABEL"
