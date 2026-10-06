#!/bin/bash
# Build + install KvCacheTurboQuant custom op inside kvtq-dev container.
set -u
docker exec -i kvtq-dev bash -s <<'EOS'
set -u
set -o pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh 2>/dev/null || source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/kv_tq/src/gen
cp /root/kv_tq/src/impl/op_host/kv_cache_turbo_quant.cpp op_host/kv_cache_turbo_quant.cpp
cp /root/kv_tq/src/impl/op_kernel/kv_cache_turbo_quant.cpp op_kernel/kv_cache_turbo_quant.cpp
cp /root/kv_tq/src/impl/op_kernel/kv_cache_turbo_quant_tiling.h op_kernel/kv_cache_turbo_quant_tiling.h
python3 - <<'PYEOF'
import json
p = "CMakePresets.json"
with open(p) as f:
    d = json.load(f)
for preset in d["configurePresets"]:
    cv = preset.get("cacheVariables", {})
    cv["ASCEND_COMPUTE_UNIT"] = {"type": "STRING", "value": "ascend910b"}
    cv["ASCEND_CANN_PACKAGE_PATH"] = {"type": "PATH", "value": "/usr/local/Ascend/cann-9.1.0"}
    cv["ASCEND_PYTHON_EXECUTABLE"] = {"type": "STRING", "value": "/usr/local/python3.12.13/bin/python3"}
with open(p, "w") as f:
    json.dump(d, f, indent=2)
print("patched presets")
PYEOF
rm -rf build_out
bash build.sh 2>&1 | tail -25
echo "=== build_out ==="
ls build_out/ 2>/dev/null
RUNFILE=$(ls build_out/*.run 2>/dev/null | head -1)
if [ -n "$RUNFILE" ]; then
  echo "=== install $RUNFILE ==="
  "$RUNFILE" --quiet --install-path=/usr/local/Ascend/cann-9.1.0/opp 2>&1 | tail -5
  echo "=== vendor tree ==="
  find /usr/local/Ascend/cann-9.1.0/opp/vendors/customize -maxdepth 3 | head -40
  echo "=== aclnn api ==="
  find /usr/local/Ascend/cann-9.1.0/opp/vendors/customize -name "*aclnn*" | head -10
fi
EOS
echo DONE