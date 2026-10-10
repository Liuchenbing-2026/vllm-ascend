#!/usr/bin/env python3
"""Fingerprint the model after benchmarking without reading weights during a cohort."""
import concurrent.futures
import datetime
import hashlib
import json
import pathlib


def hash_weight(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size, after.st_mtime_ns, after.st_ino
    ), f'Weight changed while hashing: {path.name}'
    return {'file': path.name, 'bytes': after.st_size,
            'mtime_ns': after.st_mtime_ns, 'sha256': digest.hexdigest()}


def main():
    root = pathlib.Path('/ws')
    assert (root / 'store4_high_concurrency.exit').read_text().strip() == '0'
    assert json.loads((root / 'status.json').read_text())['stage'] == 'matrix_finished'
    model = pathlib.Path('/models/Qwen3-30B-A3B')
    original = json.loads((root / 'artifacts/model-manifest.json').read_text())
    expected = {row['file']: row['bytes'] for row in original['weight_files']}
    weights = sorted(model.glob('*.safetensors'))
    assert {path.name: path.stat().st_size for path in weights} == expected
    for name, record in original['files'].items():
        assert hashlib.sha256((model / name).read_bytes()).hexdigest() == record['sha256'], name
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(hash_weight, path) for path in weights]
        for future in concurrent.futures.as_completed(tasks):
            record = future.result()
            records.append(record)
            print(json.dumps({'hashed': len(records), 'total': len(weights),
                              'file': record['file']}), flush=True)
    result = {'path': str(model), 'started_utc': started,
              'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'weight_files': sorted(records, key=lambda row: row['file']),
              'config_tokenizer_index_match_initial_manifest': True,
              'weight_names_sizes_match_initial_manifest': True,
              'scope': 'post-benchmark SHA256 snapshot; no pre/post weight hash comparison',
              'origin_revision': 'unverified'}
    target = root / 'artifacts/model-weights-sha256.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps({'complete': True, 'manifest': str(target),
                      'files': len(records)}), flush=True)


if __name__ == '__main__':
    main()
