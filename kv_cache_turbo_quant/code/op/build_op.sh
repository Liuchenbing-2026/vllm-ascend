#!/bin/bash
set -u
docker exec -i kvtq-dev2 bash -s <<'EOS'
set -uo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh 2>/dev/null
mkdir -p /root/kv_tq_i8/src
cd /root/kv_tq_i8/src
if [ ! -d gen ]; then
  msopgen gen -i KvCacheInt8Store.json -c ai_core-ascend910b -lan cpp -out gen 2>&1 | tail -5
  if [ ! -d gen ]; then
    echo MSOPGEN_FAILED
    msopgen gen -h 2>&1 | head -30
    exit 1
  fi
fi
cd gen
rm -f op_host/kv_cache_int8_store.cpp op_kernel/kv_cache_int8_store.cpp
cp /root/kv_tq_i8/src/impl/op_host/*.cpp op_host/
cp /root/kv_tq_i8/src/impl/op_kernel/*.cpp op_kernel/
cp /root/kv_tq_i8/src/impl/op_kernel/*.h op_kernel/
python3 - <<'PYEOF'
import json
p = "CMakePresets.json"
d = json.load(open(p))
for preset in d["configurePresets"]:
    cv = preset.get("cacheVariables", {})
    cv["ASCEND_COMPUTE_UNIT"] = {"type": "STRING", "value": "ascend910b"}
    cv["ASCEND_CANN_PACKAGE_PATH"] = {"type": "PATH", "value": "/usr/local/Ascend/cann-9.1.0"}
    cv["ASCEND_PYTHON_EXECUTABLE"] = {"type": "STRING", "value": "/usr/local/python3.12.13/bin/python3"}
json.dump(d, open(p, "w"), indent=2)
print("presets patched")
PYEOF
rm -rf build_out
bash build.sh > /tmp/kvtq_i8_build.log 2>&1
echo "build rc=$?"
grep -n -iE 'error|failed' /tmp/kvtq_i8_build.log | head -10
tail -3 /tmp/kvtq_i8_build.log
ls build_out/*.run 2>/dev/null
EOS
echo BUILD_SCRIPT_DONE