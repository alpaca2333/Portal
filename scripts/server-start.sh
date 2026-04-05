#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/server-common.sh"
ensure_runtime_deps
cleanup_stale_pid_file "$SUPERVISOR_PID_FILE"
cleanup_stale_pid_file "$CHILD_PID_FILE"
supervisor_pid="$(read_pid "$SUPERVISOR_PID_FILE" 2>/dev/null || true)"
if [ -n "$supervisor_pid" ] && is_pid_running "$supervisor_pid"; then
  if healthcheck; then echo "Portal backend already running (supervisor pid=$supervisor_pid)"; exit 0; fi
  echo "Existing supervisor is unhealthy, restarting (pid=$supervisor_pid)"
  "$PORTAL_ROOT/scripts/server-stop.sh" >/dev/null 2>&1 || true
fi
cd "$PORTAL_ROOT"
nohup "$PORTAL_ROOT/scripts/server-supervisor.sh" >/dev/null 2>&1 &
new_pid=$!
echo "$new_pid" > "$SUPERVISOR_PID_FILE"
echo "Starting Portal backend with supervisor pid=$new_pid"
for _ in $(seq 1 30); do
  if healthcheck; then child_pid="$(read_pid "$CHILD_PID_FILE" 2>/dev/null || true)"; echo "Portal backend is healthy (child pid=${child_pid:-unknown})"; exit 0; fi
  sleep 1
done
echo "Portal backend failed to become healthy within 30 seconds" >&2
exit 1
