#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/server-common.sh"
cleanup_stale_pid_file "$SUPERVISOR_PID_FILE"
cleanup_stale_pid_file "$CHILD_PID_FILE"

supervisor_pid="$(read_pid "$SUPERVISOR_PID_FILE" 2>/dev/null || true)"
child_pid="$(read_pid "$CHILD_PID_FILE" 2>/dev/null || true)"

for pid in "$supervisor_pid" "$child_pid" $(find_portal_pids) $(find_port_pids); do
  [ -n "${pid:-}" ] || continue
  if is_pid_running "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
done

for _ in $(seq 1 10); do
  alive=0
  for pid in "$supervisor_pid" "$child_pid" $(find_portal_pids) $(find_port_pids); do
    [ -n "${pid:-}" ] || continue
    if is_pid_running "$pid"; then
      alive=1
    fi
  done
  [ "$alive" -eq 0 ] && break
  sleep 1
done

for pid in "$supervisor_pid" "$child_pid" $(find_portal_pids) $(find_port_pids); do
  [ -n "${pid:-}" ] || continue
  if is_pid_running "$pid"; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

rm -f "$SUPERVISOR_PID_FILE" "$CHILD_PID_FILE"
echo "Portal backend stopped"
