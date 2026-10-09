#!/usr/bin/env bash
set -eo pipefail
exec > >(tee /ws/logs/framework-resume.log) 2>&1
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/ws/.venv/bin:$PATH VIRTUAL_ENV=/ws/.venv VLLM_TARGET_DEVICE=empty
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0rc1 SOC_VERSION=ascend910b4
export MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 VLLM_ASCEND_BUILD_CACHE_DIR=/ws/build-cache
export CMAKE_PREFIX_PATH="$(/ws/.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)'):${CMAKE_PREFIX_PATH:-}"
uv pip install --python /ws/.venv/bin/python -e /ws/source/vllm-ascend --no-build-isolation --no-deps -v
unset SETUPTOOLS_SCM_PRETEND_VERSION
# Keep the inherited FLA wheel out of the Ascend native build environment.
mkdir -p /root/kvtq_integration/torch_ext
cp -a /ws/source/kv_cache_turbo_quant/code/integration/*.py /root/kvtq_integration/
cp -a /ws/source/kv_cache_turbo_quant/code/integration/torch_ext/. /root/kvtq_integration/torch_ext/
cp -a /ws/source/kv_cache_turbo_quant/results/golden.py /root/kvtq_integration/
printf '/root/kvtq_integration\n' > /ws/.venv/lib/python3.12/site-packages/kvtq-path.pth
mkdir -p /ws/.venv/lib/python3.12/site-packages/kvtq_vllm_plugin-0.1.dist-info
cp /ws/source/kv_cache_turbo_quant/code/integration/{entry_points.txt,METADATA} /ws/.venv/lib/python3.12/site-packages/kvtq_vllm_plugin-0.1.dist-info/
/ws/.venv/bin/python - <<'PY'
import importlib.metadata as metadata
import json,sys,pathlib
packages={p:metadata.version(p) for p in ['vllm','vllm-ascend','torch','torch-npu','transformers','uv','setuptools']}
pathlib.Path('/ws/artifacts/packages.json').write_text(json.dumps({'python':sys.version,'packages':packages},indent=2)+'\n')
import vllm,vllm_ascend
print('SOURCE_IMPORTS',vllm.__file__,vllm_ascend.__file__)
print(packages)
PY
echo FRAMEWORK_BUILD_DONE
