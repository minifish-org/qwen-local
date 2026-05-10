#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOMAIN="gui/$(id -u)"

if [[ ! -x "$ROOT/scripts/run-server.sh" ]]; then
  echo "Missing executable server script: $ROOT/scripts/run-server.sh" >&2
  exit 1
fi

if [[ ! -x "$ROOT/scripts/run-embedding-server.sh" ]]; then
  echo "Missing executable embedding server script: $ROOT/scripts/run-embedding-server.sh" >&2
  exit 1
fi

if [[ ! -x "$ROOT/llama.cpp/build/bin/llama-server" ]]; then
  echo "llama-server is not built yet. Run ./scripts/setup-llama.sh first." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"

install_one() {
  local label="$1"
  local src="$ROOT/launchd/${label}.plist"
  local dst="$HOME/Library/LaunchAgents/${label}.plist"

  if [[ ! -f "$src" ]]; then
    echo "Missing plist template: $src" >&2
    exit 1
  fi

  cp "$src" "$dst"

  if launchctl print "$DOMAIN/$label" >/dev/null 2>&1; then
    launchctl bootout "$DOMAIN/$label" >/dev/null 2>&1 || true
  fi

  launchctl bootstrap "$DOMAIN" "$dst"
  launchctl enable "$DOMAIN/$label"
  launchctl kickstart -k "$DOMAIN/$label"

  echo "Installed and started $label"
}

install_one "org.minifish.qwen4b-local"
install_one "org.minifish.qwen4b-local-embedding"

echo "Logs:"
echo "  $ROOT/logs/launchd.out.log"
echo "  $ROOT/logs/launchd.err.log"
echo "  $ROOT/logs/launchd-embedding.out.log"
echo "  $ROOT/logs/launchd-embedding.err.log"
