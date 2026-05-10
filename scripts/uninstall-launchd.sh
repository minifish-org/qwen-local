#!/usr/bin/env bash
set -euo pipefail

DOMAIN="gui/$(id -u)"

uninstall_one() {
  local label="$1"
  local dst="$HOME/Library/LaunchAgents/${label}.plist"

  if launchctl print "$DOMAIN/$label" >/dev/null 2>&1; then
    launchctl bootout "$DOMAIN/$label"
  fi

  rm -f "$dst"
  echo "Uninstalled $label"
}

uninstall_one "org.minifish.qwen4b-local"
uninstall_one "org.minifish.qwen4b-local-embedding"
