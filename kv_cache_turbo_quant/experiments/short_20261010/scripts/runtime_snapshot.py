#!/usr/bin/env python3
"""Inspect only this container's engine processes and preserved model metadata."""
import datetime
import json
import pathlib
import subprocess


def main():
    root = pathlib.Path('/ws/short_ab_20261010')
    print('captured_utc', datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
    print('venv_python_link', pathlib.Path('/ws/.venv/bin/python').readlink(), flush=True)
    print('status', (root/'status.json').read_text(), flush=True)
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'VLLM::' not in command:
                continue
            paths = sorted({line.split()[-1] for line in (proc/'maps').read_text().splitlines()
                            if any(part in line for part in ['vllm_ascend_C', 'libvllm_ascend_kernels',
                                                             'turboquant_torch.so', 'libcust_opapi.so'])})
            print(json.dumps({'container_pid': int(proc.name), 'process': command,
                              'native_paths': paths}), flush=True)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    model = pathlib.Path('/models/Qwen3-30B-A3B')
    old = json.loads(pathlib.Path('/ws/artifacts/model-weights-sha256.json').read_text())
    current = []
    for item in old['weight_files']:
        stat = (model/item['file']).stat()
        assert stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns'], item['file']
        current.append({'file': item['file'], 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    print(json.dumps({'model_weight_names_sizes_mtimes_match_prior_post_benchmark': True,
                      'files': current, 'scope': 'metadata verification only; no new full weight hash'}), flush=True)
    subprocess.run(['npu-smi', 'info'], check=True)


if __name__ == '__main__':
    main()
