#!/usr/bin/env bash
set -euo pipefail

PORTAL_ROOT="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/..")"
RUNTIME_DIR="$PORTAL_ROOT/runtime"
APP_LOG="$RUNTIME_DIR/server.log"
SUPERVISOR_LOG="$RUNTIME_DIR/server-supervisor.log"
SUPERVISOR_PID_FILE="$RUNTIME_DIR/server-supervisor.pid"
CHILD_PID_FILE="$RUNTIME_DIR/server.pid"
HEALTH_URL="${PORTAL_HEALTH_URL:-http://127.0.0.1:8080/api/quant/status}"
NODE_BIN_DIR="${NODE_BIN_DIR:-/root/.nvm/versions/node/v24.14.1/bin}"
TSX_BIN="$PORTAL_ROOT/node_modules/.bin/tsx"
SERVER_ENTRY="$PORTAL_ROOT/src/server.ts"

mkdir -p "$RUNTIME_DIR"
if [ -d "$NODE_BIN_DIR" ]; then export PATH="$NODE_BIN_DIR:$PATH"; fi

is_pid_running() {
  local pid="${1:-}"
  [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

read_pid() {
  local file="$1"
  [ -f "$file" ] || return 1
  tr -d '[:space:]' < "$file"
}

cleanup_stale_pid_file() {
  local file="$1"
  local pid
  pid="$(read_pid "$file" 2>/dev/null || true)"
  if [ -n "$pid" ] && ! is_pid_running "$pid"; then rm -f "$file"; fi
}

healthcheck() { curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; }

ensure_runtime_deps() {
  if ! command -v node >/dev/null 2>&1; then echo "node is not available in PATH" >&2; exit 1; fi
  if [ ! -x "$TSX_BIN" ]; then echo "tsx is missing. Please run: npm install" >&2; exit 1; fi
}

find_port_pids() {
  python3 - <<'PY2'
import subprocess, re
try:
    out = subprocess.check_output(['ss','-ltnp'], text=True)
except Exception:
    out = ''
seen = set()
for line in out.splitlines():
    if ':8080' not in line:
        continue
    for pid in re.findall(r'pid=(\d+)', line):
        if pid not in seen:
            print(pid)
            seen.add(pid)
PY2
}

find_portal_pids() {
  python3 - <<'PY2'
import subprocess, os
root = '/root/Portal'
try:
    out = subprocess.check_output(['ps','-eo','pid,args'], text=True)
except Exception:
    out = ''
seen = set()
for line in out.splitlines()[1:]:
    line = line.strip()
    if not line:
        continue
    pid, *rest = line.split(' ', 1)
    args = rest[0] if rest else ''
    if root in args and ('server-supervisor.sh' in args or 'src/server.ts' in args or 'tsx' in args):
        if pid not in seen:
            print(pid)
            seen.add(pid)
PY2
}
