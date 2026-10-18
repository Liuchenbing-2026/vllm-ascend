#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; printf "PREFLIGHT_FAILED line=%s exit=%s\n" "$LINENO" "$rc"; exit "$rc"' ERR
cd /vllm-workspace/vllm-ascend
python /task4/preflight.py install
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
python /task4/preflight.py imports
exec bash /task4/launch.sh
