#!/bin/bash
set -u
mkdir -p /root/kv_tq/src
cp /root/kv_tq_xfer/KvCacheTurboQuant.json /root/kv_tq/src/ 2>/dev/null || true
docker exec -i kvtq-dev bash -s <<'EOS'
set -u
source /usr/local/Ascend/cann-9.1.0/set_env.sh 2>/dev/null || source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/kv_tq/src
rm -rf gen
mkdir -p gen
msopgen gen -i KvCacheTurboQuant.json -c ai_core-ascend910b -lan cpp -out gen 2>&1 | tail -15
echo "=== generated tree ==="
find gen -type f | sort
EOS
echo DONE