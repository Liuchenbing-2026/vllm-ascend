#!/usr/bin/env python3
"""Read the follow-up status and completed metrics without importing the engine."""
import json
import pathlib


def main():
    root = pathlib.Path('/ws/short_ab_20261010')
    for name in ['status.json', 'controller.exit']:
        path = root/name
        if path.exists():
            print(name, path.read_text(), flush=True)
    print('controller_tail', flush=True)
    if (root/'controller.log').exists():
        print('\n'.join((root/'controller.log').read_text(errors='replace').splitlines()[-12:]), flush=True)
    for path in sorted((root/'results').glob('*/*.json')):
        if path.name.endswith('.progress.json'):
            continue
        result = json.loads(path.read_text())
        print(json.dumps({'file': str(path.relative_to(root)),
                          **{k: result.get(k) for k in ['successful', 'failed', 'duration_s',
                                                        'output_throughput_tps', 'mean_ttft_s', 'mean_tpot_ms']}}), flush=True)
    for path in sorted((root/'logs').glob('*/serve_*.log')):
        print('serve_tail', str(path), flush=True)
        print('\n'.join(path.read_text(errors='replace').splitlines()[-5:]), flush=True)


if __name__ == '__main__':
    main()
