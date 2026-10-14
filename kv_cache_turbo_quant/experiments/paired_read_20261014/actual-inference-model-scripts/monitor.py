#!/usr/bin/env python3
"""Detached read-only runtime/client snapshots; never restart the benchmark."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/data02/tq-opt-20261011/paired_model_inference_20261014')
keys = ['vllm:num_requests_running', 'vllm:num_requests_waiting',
        'vllm:kv_cache_usage_perc', 'vllm:num_preemptions_total']
while True:
    record = {'utc': datetime.now(timezone.utc).isoformat()}
    try:
        record['status'] = json.loads((ROOT / 'status.json').read_text())
        record['processes'] = subprocess.check_output(['docker', 'top', 'tq-opt-20261011-model',
            '-eo', 'pid,pcpu,pmem,args'], text=True)
        record['loadavg'] = Path('/proc/loadavg').read_text().strip()
        with urllib.request.urlopen('http://127.0.0.1:18377/metrics', timeout=3) as response:
            text = response.read().decode()
        record['metrics'] = [line for line in text.splitlines() if any(line.startswith(key + '{') or line.startswith(key + ' ') for key in keys)]
    except Exception as error:
        record['observation_error'] = repr(error)
    with (ROOT / 'runtime-monitor.jsonl').open('a') as output:
        output.write(json.dumps(record) + '\n')
    if (ROOT / 'host.exit').exists():
        break
    time.sleep(2)
