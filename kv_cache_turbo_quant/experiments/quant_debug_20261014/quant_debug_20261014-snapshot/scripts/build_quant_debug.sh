#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/ws/.venv/bin:$PATH VIRTUAL_ENV=/ws/.venv
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 VLLM_PLUGINS=''
task_root=/ws/quant_debug_20261014
mkdir -p "$task_root/logs" "$task_root/op-build"
exec > >(tee "$task_root/logs/build.log") 2>&1
cd "$task_root/op-build"
if [ ! -f gen/CMakePresets.json ]; then
  msopgen gen -i "$task_root/code/op/TqQuantDebug.json" -c ai_core-ascend910b -lan cpp -out gen
fi
cp "$task_root/code/op/op_host/tq_quant_debug.cpp" gen/op_host/
cp "$task_root/code/op/op_kernel/tq_quant_debug.cpp" gen/op_kernel/
cp "$task_root/code/op/op_kernel/tq_quant_debug_tiling.h" gen/op_kernel/
/ws/.venv/bin/python - <<'PY'
import json,pathlib
p=pathlib.Path('gen/CMakePresets.json');d=json.loads(p.read_text())
for preset in d['configurePresets']:
 c=preset.setdefault('cacheVariables',{})
 for key,value,kind in [('ASCEND_COMPUTE_UNIT','ascend910b','STRING'),
   ('ASCEND_CANN_PACKAGE_PATH','/usr/local/Ascend/cann-9.1.0','PATH'),
   ('ASCEND_PYTHON_EXECUTABLE','/ws/.venv/bin/python','STRING'),('vendor_name','tqquantdebug','STRING')]:
  c[key]={'type':kind,'value':value}
p.write_text(json.dumps(d,indent=2)+'\n')
PY
cd gen
/ws/.venv/bin/python - <<'PY'
from pathlib import Path
p = Path('op_kernel/CMakeLists.txt')
s = p.read_text()
flag = 'npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g)'
if flag not in s:
    p.write_text(flag + '\n' + s)
PY
bash build.sh
runfiles=(build_out/*.run)
"${runfiles[0]}" --quiet --install-path="$task_root/opp"
export TORCH_EXTENSIONS_DIR="$task_root/torch-extensions"
/ws/.venv/bin/python "$task_root/code/integration/torch_ext/build_quant_debug.py"
