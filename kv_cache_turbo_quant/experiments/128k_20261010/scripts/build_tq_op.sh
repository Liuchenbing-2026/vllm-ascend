#!/usr/bin/env bash
set -eo pipefail
mkdir -p /ws/logs /ws/op-build
exec > >(tee /ws/logs/tq-op-build.log) 2>&1
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/ws/.venv/bin:$PATH VIRTUAL_ENV=/ws/.venv
op_source=/ws/source/kv_cache_turbo_quant/code/op
cd /ws/op-build
msopgen gen -i "$op_source/KvCacheTurboQuant.json" -c ai_core-ascend910b -lan cpp -out gen
cp "$op_source/op_host/kv_cache_turbo_quant.cpp" gen/op_host/
cp "$op_source/op_kernel/kv_cache_turbo_quant.cpp" gen/op_kernel/
cp "$op_source/op_kernel/kv_cache_turbo_quant_tiling.h" gen/op_kernel/
/ws/.venv/bin/python - <<'PY'
import json,pathlib
p=pathlib.Path('gen/CMakePresets.json')
d=json.loads(p.read_text())
for preset in d['configurePresets']:
    cv=preset.setdefault('cacheVariables',{})
    cv['ASCEND_COMPUTE_UNIT']={'type':'STRING','value':'ascend910b'}
    cv['ASCEND_CANN_PACKAGE_PATH']={'type':'PATH','value':'/usr/local/Ascend/cann-9.1.0'}
    cv['ASCEND_PYTHON_EXECUTABLE']={'type':'STRING','value':'/ws/.venv/bin/python'}
p.write_text(json.dumps(d,indent=2)+'\n')
PY
cd gen
bash build.sh
runfiles=(build_out/*.run)
"${runfiles[0]}" --quiet --install-path=/usr/local/Ascend/cann-9.1.0/opp
/ws/.venv/bin/python - <<'PY'
from pathlib import Path
p=Path('/usr/local/Ascend/cann-9.1.0/opp/vendors/config.ini')
lines=p.read_text().splitlines() if p.exists() else []
for i,line in enumerate(lines):
    if line.startswith('load_priority='):
        vendors=[v for v in line.split('=',1)[1].split(',') if v]
        if 'customize' not in vendors:vendors.append('customize')
        lines[i]='load_priority='+','.join(vendors)
        break
else:
    lines.append('load_priority=customize')
p.write_text('\n'.join(lines)+'\n')
print(p.read_text())
PY
echo TQ_OP_BUILD_DONE
