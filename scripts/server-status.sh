#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/server-common.sh"
cleanup_stale_pid_file "$SUPERVISOR_PID_FILE"
cleanup_stale_pid_file "$CHILD_PID_FILE"

supervisor_pid="$(read_pid "$SUPERVISOR_PID_FILE" 2>/dev/null || true)"
child_pid="$(read_pid "$CHILD_PID_FILE" 2>/dev/null || true)"

if [ -n "$supervisor_pid" ] && is_pid_running "$supervisor_pid"; then
  echo "supervisor: running (pid=$supervisor_pid)"
else
  echo "supervisor: stopped"
fi

if [ -n "$child_pid" ] && is_pid_running "$child_pid"; then
  echo "server(wrapper): running (pid=$child_pid)"
else
  echo "server(wrapper): stopped"
fi

actual_pids="$(find_port_pids | tr '
' ' ')"
if [ -n "$actual_pids" ]; then
  echo "server(listen:8080): running (pid=$actual_pids)"
else
  echo "server(listen:8080): stopped"
fi

if healthcheck; then
  echo "health: ok ($HEALTH_URL)"
else
  echo "health: failed ($HEALTH_URL)"
  exit 1
fi
