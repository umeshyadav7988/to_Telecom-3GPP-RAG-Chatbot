#!/usr/bin/env bash
#
# One-command launcher for local development.
#
#   ./start.sh            start backend + frontend
#   ./start.sh setup      install dependencies and build the index first
#
# Both servers run in the foreground; Ctrl-C stops them together.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
PY="$BACKEND/.venv/bin/python"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

setup() {
  command -v python3 >/dev/null || die "python3 not found"
  command -v npm >/dev/null || die "npm not found (Node 18+ required)"

  log "Creating the Python virtualenv"
  [ -d "$BACKEND/.venv" ] || python3 -m venv "$BACKEND/.venv"

  log "Installing backend dependencies"
  "$BACKEND/.venv/bin/pip" install --quiet --upgrade pip
  "$BACKEND/.venv/bin/pip" install --quiet -r "$BACKEND/requirements.txt"

  if [ ! -f "$BACKEND/.env" ]; then
    cp "$BACKEND/.env.example" "$BACKEND/.env"
    log "Created backend/.env — add your ANTHROPIC_API_KEY for generative answers"
  fi

  log "Installing frontend dependencies"
  (cd "$FRONTEND" && npm install --silent)

  log "Building the index"
  (cd "$BACKEND" && "$PY" scripts/ingest.py --stats)

  log "Setup complete. Run ./start.sh to launch."
}

run() {
  [ -x "$PY" ] || die "No virtualenv found. Run: ./start.sh setup"
  [ -d "$FRONTEND/node_modules" ] || die "Frontend deps missing. Run: ./start.sh setup"
  if [ ! -f "$BACKEND/data/index/meta.json" ]; then
    log "No index found — building it now"
    (cd "$BACKEND" && "$PY" scripts/ingest.py)
  fi

  # Make sure Ctrl-C takes the whole process group down, not just the foreground job.
  trap 'log "Shutting down"; kill 0' EXIT INT TERM

  log "Backend  → http://localhost:5001"
  (cd "$BACKEND" && "$PY" run.py) &

  log "Frontend → http://localhost:5173"
  (cd "$FRONTEND" && npm run dev) &

  wait
}

case "${1:-run}" in
  setup) setup ;;
  run)   run ;;
  *)     die "Unknown command '${1}'. Use: ./start.sh [setup|run]" ;;
esac
