#!/usr/bin/env python3
"""Read lightweight benchmark state; never alter the active server or client."""
import datetime
import json
import pathlib
import time


def main():
    root = pathlib.Path('/ws')
    while True:
        report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            state = json.loads((root / 'status.json').read_text())
            report.update({key: state.get(key) for key in ['stage', 'mode', 'concurrency']})
            path = state.get('result')
            if path:
                progress = pathlib.Path(path).with_suffix('.progress.json')
                if progress.exists():
                    data = json.loads(progress.read_text())
                    report['elapsed_s'] = round(data['elapsed_s'], 1)
                    report['requests'] = [{key: item.get(key) for key in
                                           ['id', 'status', 'events']}
                                          for item in data['requests']]
            report['completed'] = []
            for path in sorted((root / 'results').glob('matrix_*/*_in131072_out1024_c*.json')):
                if path.name.endswith('.progress.json'):
                    continue
                data = json.loads(path.read_text())
                report['completed'].append({key: data.get(key) for key in
                                            ['mode', 'concurrency', 'complete_cohort',
                                             'successful', 'failed', 'output_throughput_tps']})
            report['continuation_exit'] = {
                path.name: path.read_text().strip()
                for path in root.glob('*concurrency.exit')}
            print(json.dumps(report), flush=True)
        except (OSError, ValueError) as error:
            print(json.dumps({'utc': report['utc'], 'read_error': str(error)}), flush=True)
        time.sleep(50)


if __name__ == '__main__':
    main()
