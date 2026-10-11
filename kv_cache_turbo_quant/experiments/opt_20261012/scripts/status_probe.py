#!/usr/bin/env python3
"""Read actual status and completed formal cohorts without touching the job."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re


root = Path('/data02/tq-opt-20261011/opt_20261012')
out = {'checked_utc': datetime.now(timezone.utc).isoformat()}
for name in ['background-status.json', 'status.json', 'background.exit']:
    path = root / name
    out[name] = path.read_text() if path.exists() else None
out['formal'] = []
for directory in sorted((root / 'results').glob('*')):
    for path in sorted(directory.glob('s*_r*.json')):
        match = re.fullmatch(r's([12])_(bf16|original|optimized)_in(\d+)_out(\d+)_c(\d+)_r([1-3])\.json', path.name)
        if match is None:
            continue
        record = json.loads(path.read_text())
        output_len, concurrency = int(match.group(4)), int(match.group(5))
        complete = record['complete_cohort'] and record['successful'] == concurrency and not record['failed']
        requests = record['requests']
        out['formal'].append({'file': path.name, 'complete': complete,
            'successful': record['successful'], 'duration_s': record['duration_s'],
            'output_tps': record['successful'] * output_len / record['duration_s'] if complete else None,
            'ttft_s': sum(r['ttft_s'] for r in requests) / len(requests) if complete else None,
            'tpot_ms': sum(r['tpot_s'] for r in requests) * 1000 / len(requests) if complete else None})
print(json.dumps(out, indent=2))
