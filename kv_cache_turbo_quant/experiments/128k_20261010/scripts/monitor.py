#!/usr/bin/env python3
"""Read task state and completed cohort metrics without importing the frameworks."""
import json
import pathlib


def main():
    root = pathlib.Path('/ws')
    state = json.loads((root / 'status.json').read_text())
    report = {'state': state, 'completed': []}
    for path in sorted((root / 'results').glob('matrix_*/*_in131072_out1024_c*.json')):
        if path.name.endswith('.progress.json'):
            continue
        data = json.loads(path.read_text())
        report['completed'].append({
            'path': str(path), **{key: data[key] for key in [
                'mode', 'concurrency', 'successful', 'failed', 'duration_s',
                'output_throughput_tps', 'mean_ttft_s', 'mean_tpot_ms']}})
    result = state.get('result')
    if result:
        progress = pathlib.Path(result).with_suffix('.progress.json')
        if progress.exists():
            data = json.loads(progress.read_text())
            report['progress'] = {'elapsed_s': data['elapsed_s'],
                                  'requests': [{k: v for k, v in r.items()
                                                if k in ['id', 'status', 'events', 'last_token_s']}
                                               for r in data['requests']]}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
