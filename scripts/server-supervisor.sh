#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/server-common.sh"
ensure_runtime_deps
child_pid=""
health_failures=0
log_line(){ printf '[%s] %s
' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee -a "$SUPERVISOR_LOG"; }
stop_child(){
  local pid="${1:-$(read_pid "$CHILD_PID_FILE" 2>/dev/null || true)}"
  if [ -z "$pid" ] || ! is_pid_running "$pid"; then rm -f "$CHILD_PID_FILE"; return 0; fi
  log_line "stopping child pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do if ! is_pid_running "$pid"; then break; fi; sleep 1; done
  if is_pid_running "$pid"; then log_line "child pid=$pid did not stop in time, killing"; kill -KILL "$pid" 2>/dev/null || true; fi
  rm -f "$CHILD_PID_FILE"
}
start_child(){
  log_line "starting portal backend"
  env PATH="$PATH" "$TSX_BIN" "$SERVER_ENTRY" >>"$APP_LOG" 2>&1 &
  child_pid=$!
  echo "$child_pid" > "$CHILD_PID_FILE"
  log_line "child started pid=$child_pid"
}
cleanup(){ trap - INT TERM EXIT; stop_child "${child_pid:-}"; rm -f "$SUPERVISOR_PID_FILE"; log_line "supervisor exited"; }
trap cleanup INT TERM EXIT
cd "$PORTAL_ROOT"
echo $$ > "$SUPERVISOR_PID_FILE"
log_line "supervisor started pid=$$"
start_child
while true; do
  sleep 5
  current_pid="$(read_pid "$CHILD_PID_FILE" 2>/dev/null || true)"
  if [ -z "$current_pid" ] || ! is_pid_running "$current_pid"; then log_line "child process missing, restarting"; start_child; health_failures=0; continue; fi
  if healthcheck; then health_failures=0; continue; fi
  health_failures=$((health_failures + 1))
  log_line "healthcheck failed ($health_failures/3)"
  if [ "$health_failures" -ge 3 ]; then log_line "healthcheck failed too many times, restarting child pid=$current_pid"; stop_child "$current_pid"; start_child; health_failures=0; fi
done
