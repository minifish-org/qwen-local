#!/usr/bin/env bash
set -euo pipefail

LABEL="org.minifish.qwen4b-local"
DST="$HOME/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LABEL"
fi

rm -f "$DST"

echo "Uninstalled $LABEL"
