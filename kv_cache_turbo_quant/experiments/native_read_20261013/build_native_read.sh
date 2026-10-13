#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/ws/.venv/bin:$PATH VIRTUAL_ENV=/ws/.venv
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export VLLM_PLUGINS=''
root=/ws/native_read_20261013
mkdir -p "$root/logs" "$root/op-build"
exec > >(tee "$root/logs/build.log") 2>&1
cd "$root/op-build"
if [ ! -f gen/CMakePresets.json ]; then
  msopgen gen -i "$root/code/op/TqPagedRead.json" -c ai_core-ascend910b -lan cpp -out gen
fi
cp "$root/code/op/op_host/tq_paged_read.cpp" gen/op_host/
cp "$root/code/op/op_kernel/tq_paged_read.cpp" gen/op_kernel/
cp "$root/code/op/op_kernel/tq_paged_read_tiling.h" gen/op_kernel/
/ws/.venv/bin/python - <<'PY'
import json,pathlib
p=pathlib.Path('gen/CMakePresets.json');d=json.loads(p.read_text())
for preset in d['configurePresets']:
 c=preset.setdefault('cacheVariables',{})
 for key,value,kind in [('ASCEND_COMPUTE_UNIT','ascend910b','STRING'),
   ('ASCEND_CANN_PACKAGE_PATH','/usr/local/Ascend/cann-9.1.0','PATH'),
   ('ASCEND_PYTHON_EXECUTABLE','/ws/.venv/bin/python','STRING'),('vendor_name','tqread','STRING')]:
  c[key]={'type':kind,'value':value}
p.write_text(json.dumps(d,indent=2)+'\n')
PY
cd gen
bash build.sh
runfiles=(build_out/*.run)
"${runfiles[0]}" --quiet --install-path="$root/opp"
export TORCH_EXTENSIONS_DIR="$root/torch-extensions"
/ws/.venv/bin/python "$root/code/integration/torch_ext/build_native_read.py"
echo 0 > "$root/build.exit"
