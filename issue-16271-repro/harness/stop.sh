#!/bin/bash
# /nt/stop.sh -- stop the vllm server started by /nt/serve.sh (container-scoped only)
for p in $(pgrep -f "vllm serve /home/models/Qwen3.6-35B-A3B"); do kill "$p" 2>/dev/null; done
sleep 8
for p in $(pgrep -f "VLLM::"); do kill -9 "$p" 2>/dev/null; done
for p in $(pgrep -f "vllm serve /home/models/Qwen3.6-35B-A3B"); do kill -9 "$p" 2>/dev/null; done
sleep 5
echo "remaining:"; pgrep -af "vllm serve|VLLM::" | head
