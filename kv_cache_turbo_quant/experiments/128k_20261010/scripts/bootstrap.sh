#!/usr/bin/env bash
set -eo pipefail
mkdir -p /ws/logs /ws/source/vllm /ws/source/vllm-ascend /ws/artifacts
exec > >(tee /ws/logs/bootstrap.log) 2>&1
source /usr/local/Ascend/cann-9.1.0/set_env.sh
tar xzf /ws/vllm-source.tar.gz -C /ws/source/vllm
tar xzf /ws/ascend-source.tar.gz -C /ws/source/vllm-ascend
tar xzf /ws/tq-source.tar.gz -C /ws/source
cp -a /vllm-workspace/vllm-ascend/csrc/third_party/catlass/. /ws/source/vllm-ascend/csrc/third_party/catlass/
# Preserve the inherited registration file but disable it for the native build.
for p in /usr/local/python3.12.13/lib/python3.12/site-packages/fla_npu_opp_env.pth; do
    if [ -f "$p" ]; then mv "$p" /ws/artifacts/; fi
done
command -v uv >/dev/null || /usr/local/python3.12.13/bin/python3 -m pip install uv
uv venv --system-site-packages --python /usr/local/python3.12.13/bin/python3 /ws/.venv
export PATH=/ws/.venv/bin:$PATH VIRTUAL_ENV=/ws/.venv
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
uv pip install --python /ws/.venv/bin/python uv==0.12.24 numpy==1.26.4 ml-dtypes==0.5.3 fastokens==0.2.0 fastapi==0.136.0 starlette==1.0.1
uv pip install --python /ws/.venv/bin/python 'setuptools>=77.0.3,<81' 'setuptools-scm>=8' 'setuptools-rust>=1.9' wheel
export VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0
uv pip install --python /ws/.venv/bin/python -e /ws/source/vllm --no-build-isolation --no-deps -v
export SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0rc1 SOC_VERSION=ascend910b4
export MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 VLLM_ASCEND_BUILD_CACHE_DIR=/ws/build-cache
export CMAKE_PREFIX_PATH="$(/ws/.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)'):${CMAKE_PREFIX_PATH:-}"
uv pip install --python /ws/.venv/bin/python -e /ws/source/vllm-ascend --no-build-isolation --no-deps -v
unset SETUPTOOLS_SCM_PRETEND_VERSION
export PYTHONPATH=/ws/source/vllm:/ws/source/vllm-ascend:${PYTHONPATH:-}
# Keep the inherited FLA wheel out of the Ascend native build environment.
mkdir -p /root/kvtq_integration/torch_ext
cp -a /ws/source/kv_cache_turbo_quant/code/integration/*.py /root/kvtq_integration/
cp -a /ws/source/kv_cache_turbo_quant/code/integration/torch_ext/. /root/kvtq_integration/torch_ext/
cp -a /ws/source/kv_cache_turbo_quant/results/golden.py /root/kvtq_integration/
mkdir -p /root/kvtq_integration/golden
cp -a /ws/source/kv_cache_turbo_quant/results/golden.py /root/kvtq_integration/golden/
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
assert vllm.__file__.startswith('/ws/source/vllm/')
assert vllm_ascend.__file__.startswith('/ws/source/vllm-ascend/')
print(packages)
PY
echo FRAMEWORK_BUILD_DONE
