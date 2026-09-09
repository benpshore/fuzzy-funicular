#!/bin/sh
# Install (or reinstall) the per-user launchd agent that keeps `funicular serve --watch` running.
# Local only: no GitHub Actions, no cron, no external watchdog. Undo with: deploy/uninstall-launchd.sh
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=com.funicular.server
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
[ -f "$REPO/.env" ] || { echo "Create $REPO/.env first (see .env.example)"; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s#__REPO__#$REPO#g" -e "s#__HOME__#$HOME#g" "$REPO/deploy/$LABEL.plist" > "$PLIST"
chmod 600 "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart -k "gui/$(id -u)/$LABEL"
echo "installed $PLIST"
echo "logs: $HOME/Library/Logs/funicular.log  (tail -f)"
