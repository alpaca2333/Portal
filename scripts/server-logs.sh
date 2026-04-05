#!/usr/bin/env bash
set -euo pipefail
PORTAL_ROOT="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")"
LOG_FILE="$PORTAL_ROOT/runtime/server.log"
if [ ! -f "$LOG_FILE" ]; then echo "No log file found: $LOG_FILE"; exit 1; fi
tail -n 200 -f "$LOG_FILE"
