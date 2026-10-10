#!/usr/bin/env python3
"""Observe this follow-up without controlling or restarting the benchmark."""
import datetime
import json
import pathlib
import time


def main():
    root = pathlib.Path('/ws/short_ab_20261010')
    for _ in range(100):
        try:
            status = json.loads((root/'status.json').read_text())
            item = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'status': status}
            result = status.get('result')
            if result:
                progress = pathlib.Path(result).with_suffix('.progress.json')
                if progress.exists():
                    value = json.loads(progress.read_text())
                    events = [r['events'] for r in value['requests']]
                    item['progress'] = {'elapsed_s': value['elapsed_s'],
                                        'text_events_min': min(events),
                                        'text_events_max': max(events),
                                        'scope': 'SSE event counts, not verified token counts'}
            completed = []
            for path in sorted((root/'results').glob('*/*_r*.json')):
                if path.name.endswith('.progress.json'):
                    continue
                value = json.loads(path.read_text())
                completed.append({'file': path.name, 'success': value['successful'],
                                  'failed': value['failed'], 'duration_s': value['duration_s'],
                                  'output_tps': value['output_throughput_tps']})
            item['formal_completed'] = completed
            if (root/'controller.exit').exists():
                item['controller_exit'] = (root/'controller.exit').read_text().strip()
            print(json.dumps(item), flush=True)
            if 'controller_exit' in item:
                return
        except (FileNotFoundError, json.JSONDecodeError) as error:
            print(json.dumps({'observer_read_error': str(error)}), flush=True)
        time.sleep(45)


if __name__ == '__main__':
    main()
