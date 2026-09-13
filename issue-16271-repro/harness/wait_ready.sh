#!/bin/bash
# /nt/wait_ready.sh <tag> [timeout_s] -- block until the server answers /v1/models
TAG=${1:-run}
T=${2:-1200}
LOG=/nt/logs/serve_${TAG}.log
for i in $(seq 1 "$T"); do
  if curl -s -m 3 http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
    echo "READY after ${i}s"
    exit 0
  fi
  if grep -qE "Traceback|Engine core initialization failed|ERROR .*died" "$LOG" 2>/dev/null; then
    echo "FAILED - tail:"
    tail -40 "$LOG"
    exit 1
  fi
  sleep 1
done
echo "TIMEOUT"
tail -30 "$LOG"
exit 1
