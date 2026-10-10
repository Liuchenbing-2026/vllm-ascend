#!/usr/bin/env python3
"""Read shared-host pressure without inspecting other jobs' arguments."""
import datetime
import json
import os
import pathlib
import time


def cpu_times():
    line = pathlib.Path('/proc/stat').read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in line]
    # guest/guest_nice are already included in user/nice.
    return sum(values[:8]), values[3] + values[4]


def main():
    before = cpu_times()
    time.sleep(1)
    after = cpu_times()
    memory = {}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ['MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree']:
            memory[key + '_kib'] = int(value.split()[0])
    elapsed = after[0] - before[0]
    print(json.dumps({
        'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'logical_cpu_count': os.cpu_count(),
        'load_average': list(os.getloadavg()),
        'cpu_busy_fraction_one_second': 1 - (after[1] - before[1]) / elapsed,
        'memory': memory,
        'scope': 'one shared-host snapshot; no proof of absence of interference',
    }, indent=2))


if __name__ == '__main__':
    main()
