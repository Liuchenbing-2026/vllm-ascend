#!/usr/bin/env python3
"""Copy explicitly selected cohorts; preserve their source paths and hashes."""
import argparse
import hashlib
import json
import pathlib

from summarize_results import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True, type=pathlib.Path)
    parser.add_argument('--selection', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    selection = json.loads(args.selection.read_text())
    records = []
    seen = set()
    for relative in selection:
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError(f'Selected source escapes source root: {relative}')
        data = validate(path)
        key = (data['mode'], data['concurrency'])
        if key in seen:
            raise ValueError(f'Multiple selected cohorts for {key}')
        if key[0] not in ['bf16', 'store4'] or key[1] not in [1, 2, 4, 8, 16, 32]:
            raise ValueError(f'Unexpected selected cohort: {key}')
        seen.add(key)
        payload = path.read_bytes()
        destination = args.output / path.name
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f'Refusing to overwrite different cohort: {destination}')
        records.append({'source_path': relative, 'selected_path': path.name,
                        'sha256': hashlib.sha256(payload).hexdigest(),
                        'mode': key[0], 'concurrency': key[1],
                        'complete_cohort': data['complete_cohort']})
    args.output.mkdir(parents=True, exist_ok=True)
    for record in records:
        (args.output / record['selected_path']).write_bytes(
            (source_root / record['source_path']).read_bytes())
    (args.output / 'selection-provenance.json').write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + '\n')
    print(f'Validated and selected {len(records)} cohorts')


if __name__ == '__main__':
    main()
