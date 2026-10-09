#!/usr/bin/env python3
"""Capture the actual task environment, import paths and native artifact hashes."""
import hashlib
import importlib.metadata as metadata
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone


def main():
    root = pathlib.Path('/ws')
    import vllm
    import vllm_ascend
    import numpy
    assert vllm.__file__.startswith('/ws/source/vllm/')
    assert vllm_ascend.__file__.startswith('/ws/source/vllm-ascend/')
    names = ['vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers',
             'numpy', 'ml-dtypes', 'fastapi', 'starlette', 'fastokens',
             'pydantic', 'pydantic-core', 'aiohttp', 'uv', 'setuptools',
             'setuptools-scm', 'cmake', 'pybind11', 'triton-ascend']
    packages = {}
    for name in names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    paths = [root / 'source/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so',
             root / 'source/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so',
             pathlib.Path('/usr/local/Ascend/cann-9.1.0/opp/vendors/customize/op_api/lib/libcust_opapi.so')]
    paths.extend((root / 'torch-extensions').glob('*/*.so'))
    artifacts = {str(path): {'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in paths if path.exists()}
    output = {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'python_executable': sys.executable, 'packages': packages,
        'imports': {'vllm': vllm.__file__, 'vllm_version': vllm.__version__,
                    'vllm_ascend': vllm_ascend.__file__, 'numpy': numpy.__file__},
        'native_artifacts': artifacts,
        'catlass_commit': subprocess.check_output([
            'git', '-C', '/vllm-workspace/vllm-ascend/csrc/third_party/catlass',
            'rev-parse', 'HEAD'], text=True).strip(),
    }
    (root / 'artifacts/environment.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
