#!/usr/bin/env bash
# start_all.sh — Start all IR system services locally (no Docker).
#
# Usage:  bash scripts/start_all.sh
#         LOG_DIR=logs bash scripts/start_all.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

# ── Service definitions: (name port module directory) ─────────────────────────
declare -a SERVICES=(
  "preprocessing   8001 main $ROOT/services/preprocessing"
  "indexing        8002 main $ROOT/services/indexing"
  "retrieval       8003 main $ROOT/services/retrieval"
  "query_refinement 8005 main $ROOT/services/query_refinement"
  "evaluation      8006 main $ROOT/services/evaluation"
  "api_gateway     8000 main $ROOT/services/api_gateway"
)

# ── Helper: wait for a service's /health endpoint ────────────────────────────
wait_healthy() {
  local name="$1"
  local port="$2"
  local max_attempts=30
  local attempt=0

  echo -n "  Waiting for $name (port $port)…"
  until curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
      echo " TIMEOUT"
      echo "ERROR: $name did not become healthy after $max_attempts seconds." >&2
      exit 1
    fi
    sleep 1
    echo -n "."
  done
  echo " OK"
}

# ── Start each service ────────────────────────────────────────────────────────
echo "Starting IR System services…"
echo

for entry in "${SERVICES[@]}"; do
  read -r name port module svc_dir <<< "$entry"
  log_file="$LOG_DIR/$name.log"

  echo "Starting $name on port $port…"
  (
    cd "$svc_dir"
    uvicorn "$module:app" --host 0.0.0.0 --port "$port" --reload \
      >> "$log_file" 2>&1 &
    echo $! > "$LOG_DIR/$name.pid"
  )
done

echo
echo "Waiting for services to become healthy…"
echo

for entry in "${SERVICES[@]}"; do
  read -r name port module svc_dir <<< "$entry"
  wait_healthy "$name" "$port"
done

echo
echo "All services running."
echo
echo "  API Gateway:      http://localhost:8000"
echo "  API Docs:         http://localhost:8000/docs"
echo "  Preprocessing:    http://localhost:8001"
echo "  Indexing:         http://localhost:8002"
echo "  Retrieval:        http://localhost:8003"
echo "  Query Refinement: http://localhost:8005"
echo "  Evaluation:       http://localhost:8006"
echo
echo "Logs in: $LOG_DIR"
echo "To stop: kill \$(cat $LOG_DIR/*.pid)"
