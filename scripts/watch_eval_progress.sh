#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <container_name> [log_file]" >&2
  exit 1
fi

CONTAINER_NAME="$1"
LOG_FILE="${2:-$(pwd)/logs/${CONTAINER_NAME}_progress.log}"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[watch] container=${CONTAINER_NAME}"
echo "[watch] log=${LOG_FILE}"

docker logs -f "$CONTAINER_NAME" 2>&1 | tee "$LOG_FILE" &
LOGS_PID=$!

cleanup() {
  if kill -0 "$LOGS_PID" 2>/dev/null; then
    kill "$LOGS_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

last_size=0
last_change=$(date +%s)

while sleep 60; do
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "[watch] container stopped: ${CONTAINER_NAME}"
    exit 0
  fi

  size=$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)
  now=$(date +%s)

  if [[ "$size" -gt "$last_size" ]]; then
    last_size="$size"
    last_change="$now"
    echo "[watch] log advanced: size=${size}"
    continue
  fi

  idle=$((now - last_change))
  echo "[watch] no log growth for ${idle}s"

  if [[ "$idle" -ge 1800 ]]; then
    echo "[watch] stopping ${CONTAINER_NAME} after 30 minutes without log growth"
    docker stop "$CONTAINER_NAME" >/dev/null
    exit 0
  fi
done